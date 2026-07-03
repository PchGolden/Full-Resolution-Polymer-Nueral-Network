"""Refactored token?pair joint encoder (12?layer Transformer derivative)."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import Attention
from .utils import DropPath, OuterProduct, TriangleMultiplication, Transition


class EncoderBlock(nn.Module):
    """Stacked *EncoderLayer*s with pre?norm and pair pathway."""

    def __init__(
        self,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        *,
        pair_dim: int = 128,
        pair_hidden_dim: int = 32,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        droppath_prob: float = 0.0,
        pair_dropout: float = 0.25,
        wo_triopm: bool = False,
        wo_pair: bool = False,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.pair_layer_norm = nn.LayerNorm(pair_dim)

        # DropPath schedule (stochastic depth)
        droppath_probs = (
            torch.linspace(0, droppath_prob, num_encoder_layers).tolist()
            if droppath_prob > 0
            else None
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    embedding_dim=embedding_dim,
                    pair_dim=pair_dim,
                    pair_hidden_dim=pair_hidden_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    droppath_prob=droppath_probs[i] if droppath_probs else 0.0,
                    pair_dropout=pair_dropout,
                    wo_triopm=wo_triopm,
                    wo_pair=wo_pair,
                )
                for i in range(num_encoder_layers)
            ]
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        node_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        *,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-norm; iterate over *EncoderLayer*s."""
        node_repr = self.layer_norm(node_repr)
        pair_repr = self.pair_layer_norm(pair_repr)

        if atom_mask is None:
            mask_bool = None
            op_mask = None
            op_norm = None
            scale = None
        else:
            mask_bool = atom_mask.bool()
            n_valid = mask_bool.sum(-1, keepdim=True)
            scale = n_valid.clamp(min=1).float().pow(-0.5)
            op_mask = mask_bool.unsqueeze(-1).float()
            op_norm = scale.unsqueeze(-1).unsqueeze(-1)
            scale = scale[..., None]
            n_valid = mask_bool.sum(-1, keepdim=True)
            scale = n_valid.clamp(min=1).float().pow(-0.5)
            op_mask = mask_bool.unsqueeze(-1).float()
            #op_norm = scale[..., None, None]
            op_norm = scale.unsqueeze(-1).unsqueeze(-1)
            pair_mask_scaled = pair_mask            
            scale = scale[..., None]                

        for layer in self.layers:
            node_repr, pair_repr = layer(
                node_repr,
                pair_repr,
                pair_mask=pair_mask,
                self_attn_mask=attn_mask,
                op_mask=op_mask,
                op_norm=op_norm,
                scale=scale,
            )
        return node_repr, pair_repr


class EncoderLayer(nn.Module):
    """Single Transformer block with pair pathway (O(n2) memory)."""

    def __init__(
        self,
        embedding_dim: int = 768,
        *,
        pair_dim: int = 64,
        pair_hidden_dim: int = 32,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "gelu",
        droppath_prob: float = 0.0,
        pair_dropout: float = 0.25,
        wo_triopm: bool = False,
        wo_pair: bool = False,
    ) -> None:
        super().__init__()
        
        self.wo_triopm = wo_triopm
        self.wo_pair = wo_pair
        self.dropout_module = DropPath(droppath_prob) if droppath_prob else nn.Dropout(dropout)

        head_dim = embedding_dim // num_attention_heads
        self.self_attn = Attention(
            embedding_dim,
            embedding_dim,
            embedding_dim,
            pair_dim=pair_dim,
            head_dim=head_dim,
            num_heads=num_attention_heads,
            gating=True,
            dropout=attention_dropout,
            wo_pair=wo_pair,
        )

        # layer norms & projections
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.x_layer_norm_opm = nn.LayerNorm(embedding_dim)

        self.ffn = Transition(embedding_dim, ffn_embedding_dim // embedding_dim, dropout=activation_dropout)
        self.opm = OuterProduct(embedding_dim, pair_dim, d_hid=pair_hidden_dim)

        self.pair_layer_norm_trimul = nn.LayerNorm(pair_dim)
        self.pair_tri_mul = TriangleMultiplication(pair_dim, pair_hidden_dim)

        self.pair_layer_norm_ffn = nn.LayerNorm(pair_dim)
        self.pair_ffn = Transition(pair_dim, 1, dropout=activation_dropout)
        self.pair_dropout = nn.Dropout(pair_dropout)

    # ------------------------------------------------------------------
    def forward(
        self,
        node_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        *,
        pair_mask: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        op_mask: Optional[torch.Tensor] = None,
        op_norm: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---- self-attention on tokens -----------------------------------
        res = node_repr
        qry = self.self_attn_layer_norm(node_repr)
        node_repr = self.self_attn(qry, qry, qry, pair=pair_repr, mask=self_attn_mask)
        node_repr = res + self.dropout_module(node_repr)

        # ---- position-wise FFN -----------------------------------------
        node_repr = node_repr + self.dropout_module(self.ffn(self.final_layer_norm(node_repr)))
        
        
        if self.wo_triopm or self.wo_pair:
            pass
        else:        
            # ---- outer product mixing --------------------------------------
            pair_repr = pair_repr + self.dropout_module(
                self.opm(self.x_layer_norm_opm(node_repr), op_mask, op_norm)
            )
    
            # ---- triangle multiplication -----------------------------------
            pair_repr = pair_repr + self.pair_dropout(
                self.pair_tri_mul(self.pair_layer_norm_trimul(pair_repr), pair_mask, scale)
            )

        # ---- pair-wise FFN ---------------------------------------------
        pair_repr = pair_repr + self.dropout_module(
            self.pair_ffn(self.pair_layer_norm_ffn(pair_repr))
        )
        return node_repr, pair_repr
