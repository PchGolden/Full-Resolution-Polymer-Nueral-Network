# ------------------------------------------------------------------
# File: attention.py         (drop-in replacement)
# ------------------------------------------------------------------
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

_TRUNC_STD_FACTOR = 0.87962566103423978  # from scipy.stats truncnorm(-2, 2).std()


def _trunc_normal_(tensor: torch.Tensor, scale: float = 1.0) -> None:
    fan_in = tensor.shape[1]
    std = (scale / max(1, fan_in)) ** 0.5 / _TRUNC_STD_FACTOR
    nn.init.trunc_normal_(tensor, mean=0.0, std=std)


def _init_linear(layer: nn.Linear, init_type: str, use_bias: bool = True) -> None:
    if init_type == "glorot":
        nn.init.xavier_uniform_(layer.weight, gain=1.0)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    elif init_type == "gating":  # weight=0, bias=1
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.ones_(layer.bias)

    elif init_type == "final":  # weight=0, bias=0
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    else:  # "default" = truncated normal
        _trunc_normal_(layer.weight, scale=1.0)
        if layer.bias is not None and use_bias:
            nn.init.zeros_(layer.bias)

class Attention(nn.Module):
    """
    Multi-head attention with optional gating and pairwise bias,
    refactored to eliminate external dependencies.
    """

    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        pair_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
        dropout: float = 0.0,
        wo_pair: bool = False,
    ):
        super().__init__()
        self.wo_pair = wo_pair
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gating = gating
        self.dropout = dropout
        total_dim = head_dim * num_heads
        norm = head_dim ** -0.5
        self.register_buffer("_norm", torch.tensor(norm), persistent=False)

        # ---- projection layers ----
        self.linear_q = nn.Linear(q_dim, total_dim, bias=False)
        self.linear_k = nn.Linear(k_dim, total_dim, bias=False)
        self.linear_v = nn.Linear(v_dim, total_dim, bias=False)
        self.linear_o = nn.Linear(total_dim, q_dim, bias=True)
        self.linear_g = nn.Linear(q_dim, total_dim, bias=True) if gating else None
        self.linear_bias = nn.Linear(pair_dim, num_heads, bias=True)
        self.last_attn = None
        # ---- initialization (mirrors legacy implementation) ----
        _init_linear(self.linear_q, "glorot", use_bias=False)
        _init_linear(self.linear_k, "glorot", use_bias=False)
        _init_linear(self.linear_v, "glorot", use_bias=False)
        _init_linear(self.linear_o, "final")
        if self.linear_g is not None:
            _init_linear(self.linear_g, "gating")
        _init_linear(self.linear_bias, "default")

    # ------------------------------------------------------------------ #
    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x.view(*x.shape[:-1], self.num_heads, self.head_dim)
            .transpose(-2, -3)
            .contiguous()
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q, k, v: [B, N, C]
            pair: [B, N, N, pair_dim]
        """
        g = self.linear_g(q) if self.linear_g is not None else None

        # --- project & scale ---
        q = self._shape(self.linear_q(q)) * self._norm
        k = self._shape(self.linear_k(k))
        v = self._shape(self.linear_v(v))

        # --- compute attention scores ---
        attn = torch.matmul(q, k.transpose(-1, -2))  # [B, H, N, N]

        bias = self.linear_bias(pair)                # [B, N, N, H]
        bias = bias.permute(0, 3, 1, 2).contiguous()  # [B, H, N, N]
        
        if mask is not None:
            attn = attn + mask
        if self.wo_pair:
            pass
        else:
            attn = attn + bias
        attn = F.softmax(attn, dim=-1)
        self.last_attn = attn.detach()
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        o = torch.matmul(attn, v)  # [B, H, N, d]

        # --- merge heads ---
        o = o.transpose(-2, -3).contiguous()
        o = o.view(o.shape[0], o.shape[1], self.num_heads * self.head_dim)  # [B, N, C]

        if g is not None:
            o = torch.sigmoid(g) * o

        return self.linear_o(o)
