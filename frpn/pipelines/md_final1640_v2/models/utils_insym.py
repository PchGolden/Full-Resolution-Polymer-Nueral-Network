import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional

############## monomer-level utils here################

class AtomFeaturePlus(nn.Module):

    def __init__(
        self,
        num_atom: int,
        num_degree: int,
        hidden_dim: int,
        wo_node: bool = False,
        wo_atom_feat: Optional[list[int]] = None,
        num_glob_feat: int = 2,
        num_seg_feat: int = 2,
    ):
        super().__init__()
        # --- Same three types of embeddings as the official version ---
        self.atom_encoder   = Embedding(num_atom,   hidden_dim, padding_idx=0)
        self.degree_encoder = Embedding(num_degree, hidden_dim, padding_idx=0)
        self.vnode_encoder  = Embedding(1,          hidden_dim)  # [CLS] token
        self.wo_node = wo_node
        self.wo_atom_feat = wo_atom_feat
        # --- Additional: projection layers for numeric global/segment features ---
        self.num_glob_feat = num_glob_feat
        self.num_seg_feat  = num_seg_feat
        self.glob_projs = nn.ModuleList(
            [Linear(1, hidden_dim, bias=True, init="glorot") for _ in range(num_glob_feat)]
        )
        self.seg_projs = nn.ModuleList(
            [Linear(1, hidden_dim, bias=True, init="glorot") for _ in range(num_seg_feat)]
        )

    # ------------------------------ forward ------------------------------
    def forward(self, batched_data, token_feat):
    
        x           = batched_data["atom_feat"]          # [B, N, F_a]
        degree      = batched_data["degree"]             # [B, N]
        seg_id      = batched_data["segment_id"]         # [B, N]
        glob_f      = batched_data["glob_feat"]          # [B, G]
        glob_mask   = batched_data["glob_mask"]          # [B, G]
        glob_valid  = batched_data["glob_valid_mask"]
        seg_f       = batched_data["seg_feat"]           # [B, S, F_s]
        seg_f_mask  = batched_data["seg_feat_mask"]      # [B, S, F_s]
        seg_valid   = batched_data["seg_valid_mask"]     # [B, S]
    
        B, N_atom = x.shape[:2]
        S         = seg_f.size(1)
        D         = token_feat.size(-1)
        dtype     = token_feat.dtype
        device    = x.device
    
        # ---------- 1. Atom nodes ----------
        if self.wo_node:
            atom_vec = token_feat
        else:
            atom_emb = self.atom_encoder(x)
        
            if self.wo_atom_feat is not None:
                atom_emb[:, :, self.wo_atom_feat, :] = 0.0
            atom_vec = atom_emb.sum(dim=-2)
        
            if self.wo_atom_feat is None or 1 not in self.wo_atom_feat:
                atom_vec = atom_emb.sum(dim=-2) + self.degree_encoder(degree)
        
            atom_vec = atom_vec + token_feat
    
        # ---------- 2. GLOB token ----------
        glob_vec = torch.zeros(B, D, device=device)
        

        for i in range(self.num_glob_feat):
            val  = glob_f[:, i].unsqueeze(-1)            # [B,1]
            gmsk = glob_mask[:, i].unsqueeze(-1)         # [B,1]
            glob_vec += gmsk * self.glob_projs[i](val)   
        # glob_vec=0 if entire row is None.
        glob_vec = glob_vec * glob_valid
        glob_vec = glob_vec.unsqueeze(1)  
        # ---------- 3. SEG tokens ----------
        seg_vec = torch.zeros(B, S, D, device=device)
        for j in range(self.num_seg_feat):
            val  = seg_f[:, :, j].unsqueeze(-1)          # [B,S,1]
            smsk = seg_f_mask[:, :, j].unsqueeze(-1)     # [B,S,1]
            seg_vec += smsk * self.seg_projs[j](val)
        # seg_valid
        seg_vec = seg_vec * seg_valid.unsqueeze(-1)      # [B,S,D]
    
        # ---------- 4. CLS ----------
        cls_vec = self.vnode_encoder.weight.unsqueeze(0).repeat(B, 1, 1)
    
        # ---------- 5. Concat ----------
        graph_node_feature = torch.cat(
            [cls_vec, glob_vec, seg_vec, atom_vec], dim=1
        ).type(dtype)                                    # [B, 1+1+S+N, D]
        return graph_node_feature
        

class EdgeFeaturePlus(nn.Module):

    def __init__(self, pair_dim, num_edge, num_spatial, wo_spd, wo_edge):
        super().__init__()
        self.pair_dim = pair_dim
        self.edge_encoder       = Embedding(num_edge,    pair_dim, padding_idx=0)
        self.shorest_path_encoder  = Embedding(num_spatial, pair_dim, padding_idx=0)
        self.vnode_virtual_distance = Embedding(1,           pair_dim)  # virtual bias t
        self.wo_spd = wo_spd
        self.wo_edge = wo_edge

    # ------------------------------------------------------------------
    def forward(self, batched_data, graph_attn_bias):

        shortest_path = batched_data["shortest_path"]          # [B,N,N]
        edge_input    = batched_data["edge_feat"]              # [B,N,N,K]
        n_seg         = batched_data["seg_feat"].size(1)
        N_atom        = shortest_path.size(-1)
        special_len   = 1 + 1 + n_seg                          # CLS+GLOB+SEG
        B             = graph_attn_bias.size(0)

        # ------ 1. atom - atom bias: shortest path + edge features ------
        if self.wo_spd and self.wo_edge:
            pass
        else:
            if self.wo_spd:
                atom_bias = self.edge_encoder(edge_input).mean(-2)     # [B,N,N,D]
            elif self.wo_edge:
                atom_bias = self.shorest_path_encoder(shortest_path)   # [B,N,N,D]
            else:
                atom_bias = self.shorest_path_encoder(shortest_path) + self.edge_encoder(edge_input).mean(-2)     # [B,N,N,D]            
            graph_attn_bias[:, special_len:, special_len:, :] = atom_bias

        # ------ 2. Any pair involving a special token -> use t bias ------
        t = self.vnode_virtual_distance.weight.view(1, 1, self.pair_dim)
        graph_attn_bias[:, :special_len, :, :] = t             # row
        graph_attn_bias[:, :, :special_len, :] = t             # col
        
        return graph_attn_bias
        
        
def _build_attn_mask(base_mask, seg_id, atom_mask, seg_valid, glob_valid, *, num_heads: int):
    """Construct the *additive* attention mask (-inf for disallowed positions).

    Args:
        base_mask (Tensor): user-provided base mask, shape [B, T, T]
        seg_id (Tensor): segment ids for each atom [B, N]
        atom_mask (Tensor): padding mask for atoms [B, N]
        seg_valid (Tensor): which segments are valid [B, S]
        glob_valid (Tensor): [B, 1] indicating if GLOB token is valid
        num_heads (int): number of attention heads (for expansion)
    
    Returns:
        Tensor: [B, num_heads, T, T] additive mask (0 for allowed, -inf for block)
    """
    B, S = seg_valid.shape
    N = seg_id.shape[1]

    special_len = 2 + S  # CLS + GLOB + SEG
    T = special_len + N
    device = base_mask.device

    attn_mask = base_mask.clone()
    padding_msk = attn_mask == float("-inf")

    seg_rows_all = torch.arange(S, device=device) + 2
    atom_rows = torch.arange(N, device=device) + special_len

    for b in range(B):
        seg_rows = seg_rows_all[seg_valid[b] == 1]

        # 1. Mask all segments by default
        attn_mask[b, seg_rows_all, :] = float("-inf")
        attn_mask[b, :, seg_rows_all] = float("-inf")

        # 2. Allow CLS <-> valid segments
        attn_mask[b, seg_rows, 0] = 0
        attn_mask[b, 0, seg_rows] = 0

        # 3. Allow GLOB <-> valid segments only if glob_valid[b] == 1
        if glob_valid[b] == 1:
            attn_mask[b, seg_rows, 1] = 0
            attn_mask[b, 1, seg_rows] = 0
        else:
            # Block GLOB token completely
            attn_mask[b, 1, :] = float("-inf")
            attn_mask[b, :, 1] = float("-inf")

        # 4. segment <-> own atoms
        for idx_s, row in zip(torch.nonzero(seg_valid[b]).flatten(), seg_rows):
            idx_atoms = atom_rows[seg_id[b] == idx_s]
            attn_mask[b, row, idx_atoms] = 0
            attn_mask[b, idx_atoms, row] = 0
            attn_mask[b, row, row] = 0  # self

        # 5. padding atoms (block both row and column)
        pad_atoms = atom_rows[atom_mask[b] == 0]
        attn_mask[b, pad_atoms, :] = float("-inf")
        attn_mask[b, :, pad_atoms] = float("-inf")

    # 6. Ensure diagonal is 0 (self-attention allowed unless already -inf)
    diag_idx = torch.arange(T, device=device)
    attn_mask[:, diag_idx, diag_idx] = 0.0

    return attn_mask.unsqueeze(1).float()  # [B, num_heads, T, T]
    
    
def build_padding_only_attn_mask(atom_mask, seg_valid, glob_valid, *,
                                 num_heads: int, device=None):
    B, N = atom_mask.shape
    S = seg_valid.shape[1]
    device = atom_mask.device if device is None else device

    cls_valid  = torch.ones(B, 1, dtype=torch.bool, device=device)          # CLS=1
    glob_valid = glob_valid.view(B, 1).bool().to(device)                    # GLOB��{0,1}
    seg_valid  = seg_valid.bool().to(device)                                # S seg tokens
    atom_valid = atom_mask.bool().to(device)                                # N atom tokens

    token_valid = torch.cat([cls_valid, glob_valid, seg_valid, atom_valid], dim=1)
    T = token_valid.size(1)

    valid_pair = token_valid.unsqueeze(-1) & token_valid.unsqueeze(-2)      # outer AND

    attn = torch.zeros(B, 1, T, T, device=device, dtype=torch.float32)
    attn = attn.masked_fill(~valid_pair.unsqueeze(1), float("-inf"))
    
    eye = torch.eye(T, dtype=torch.bool, device=device).view(1, 1, T, T)
    attn = torch.where(eye, torch.zeros_like(attn), attn)

    attn = attn.expand(-1, num_heads, -1, -1).contiguous()
    return attn
    
############## monomer-level utils here################    


############## Chain-level utils here################

class ChainTokenFeaturePlus(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_glob_feat: int,
        num_block_feat: int,
        enable_topology_symmetry_break: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_glob_feat = num_glob_feat
        self.num_block_feat = num_block_feat
        self.enable_topology_symmetry_break = bool(enable_topology_symmetry_break)

        # ---- CLS token ----
        self.chain_cls = nn.Embedding(1, embed_dim)

        # ---- Global feature projections (like AtomFeaturePlus) ----
        self.glob_projs = nn.ModuleList(
            [nn.Linear(1, embed_dim, bias=True) for _ in range(num_glob_feat)]
        )

        # ---- Block feature projections (shared across blocks) ----
        self.block_projs = nn.ModuleList(
            [nn.Linear(1, embed_dim, bias=True) for _ in range(num_block_feat)]
        )
        
        # Legacy (AB junction) symmetry breaking: scalar signed distance to junction.
        self.topo_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Topology-driven symmetry breaking: use distances to two anchors.
        # Kept separate to avoid breaking legacy checkpoint shapes.
        self.topo_mlp_anchor2 = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.topo_bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(
        self,
        seg_repr,          # [B, S, D]
        seg_dop,           # [B, S]
        seg_block_id,      # [B, S]
        glob_feat,         # [B, G]
        glob_feat_mask,    # [B, G]
        block_feat,        # [B, Bmax, F]
        block_feat_mask,   # [B, Bmax, F]
        chain_node_seg_id=None,
        chain_topo_dist=None,
        num_heads: int = 8,
    ):
        """
        Returns:
            chain_tokens : [B, T, D]
            attn_mask    : [B, H, T, T]
        """
        B, S, D = seg_repr.shape
        device = seg_repr.device

        # number of block tokens is determined by preprocessing (Bmax can be 0)
        if block_feat is None or block_feat_mask is None:
            Bmax = 0
        else:
            if not isinstance(block_feat, torch.Tensor) or not isinstance(block_feat_mask, torch.Tensor):
                raise TypeError("block_feat and block_feat_mask must be torch.Tensor when provided")
            if block_feat.dim() != 3 or block_feat_mask.dim() != 3:
                raise ValueError(f"block_feat/block_feat_mask must be 3D, got {block_feat.shape} and {block_feat_mask.shape}")
            Bmax = int(block_feat.size(1))

        chain_tokens_list = []
        chain_block_list = []  # legacy only
        chain_len_list = []
        topo_dist_list = []
        block_valid_list = []

        def _as_1d_long(x, device: torch.device) -> torch.Tensor:
            if x is None:
                return torch.empty(0, dtype=torch.long, device=device)
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=torch.long).view(-1)
            # python list / numpy
            return torch.tensor(list(x), dtype=torch.long, device=device).view(-1)

        def _as_2d_long(x, device: torch.device) -> torch.Tensor:
            if x is None:
                return torch.empty(0, 0, dtype=torch.long, device=device)
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=torch.long)
            return torch.tensor(x, dtype=torch.long, device=device)

        # ---------- build unpadded sequences ----------
        for b in range(B):
            tokens = []
            blocks = []

            # CLS
            cls_tok = self.chain_cls.weight.unsqueeze(0)   # [1, 1, D]
            cls_tok = cls_tok.repeat(1, 1, 1).squeeze(0)   # [1, D]
            tokens.append(cls_tok)
            blocks.append(-1)

            # GLOB
            glob_vec = torch.zeros(D, device=device)
            
            # glob_feat[b] : Tensor [G]
            # glob_feat_mask[b] : Tensor [G]
            feat_b = glob_feat[b]
            mask_b = glob_feat_mask[b]
            
            for i in range(self.num_glob_feat):
                if mask_b[i].item() == 1:
                    glob_vec += self.glob_projs[i](feat_b[i].view(1, 1)).view(-1)
            
            tokens.append(glob_vec.view(1, D))
            blocks.append(-1)

            # BLOCK tokens (variable count, pad-able like stage-1 SEG)
            block_valid = torch.zeros(Bmax, dtype=torch.bool, device=device)
            if Bmax > 0:
                block_feat_b = block_feat[b]                 # [Bmax, F]
                block_feat_mask_b = block_feat_mask[b]       # [Bmax, F]
                block_valid = (block_feat_mask_b.sum(dim=-1) > 0)

                for blk in range(Bmax):
                    blk_vec = torch.zeros(D, device=device)
                    for j in range(self.num_block_feat):
                        if j >= int(block_feat_b.size(1)):
                            break
                        if block_feat_mask_b[blk, j].item() == 1:
                            blk_vec += self.block_projs[j](block_feat_b[blk, j].view(1, 1)).view(-1)
                    # if a block is invalid, keep it zero; masking happens in attn_mask
                    if not bool(block_valid[blk].item()):
                        blk_vec.zero_()
                    tokens.append(blk_vec.view(1, D))
                    blocks.append(blk)

            # =========================
            # New path: topology-driven node sequence
            # =========================
            if chain_node_seg_id is not None:
                node_seg = chain_node_seg_id[b] if isinstance(chain_node_seg_id, (list, tuple)) else chain_node_seg_id
                node_seg = _as_1d_long(node_seg, device)
                L = int(node_seg.numel())

                if L > 0:
                    node_tok = seg_repr[b].index_select(0, node_seg)  # [L, D]
                    tokens.append(node_tok)

                # Symmetry breaking (topology-derived) — optional
                topo = None
                if chain_topo_dist is not None:
                    topo_b = chain_topo_dist[b] if isinstance(chain_topo_dist, (list, tuple)) else chain_topo_dist
                    topo = _as_2d_long(topo_b, device)
                    if (
                        self.enable_topology_symmetry_break
                        and topo.dim() == 2
                        and topo.size(0) == L
                        and topo.size(1) == L
                        and L > 0
                    ):
                        dist_sum = topo.float().sum(dim=1)
                        anchor1 = int(torch.argmin(dist_sum).item())
                        d_from_a1 = topo[anchor1].float()
                        anchor2 = int(torch.argmax(d_from_a1).item())

                        d1 = topo[:, anchor1].float()
                        d2 = topo[:, anchor2].float()
                        denom = torch.maximum(d1.max(), d2.max()).clamp(min=1.0)
                        topo_feat = torch.stack([d1 / denom, d2 / denom], dim=-1)  # [L, 2]

                        topo_field = self.topo_mlp_anchor2(topo_feat)  # [L, D]
                        rewrite_mask = torch.ones(L, 1, device=device)
                        # Apply only to node tokens (after CLS/GLOB)
                        if L > 0:
                            tokens[-1] = tokens[-1] * (1.0 + topo_field) + self.topo_bias * rewrite_mask

                tokens = torch.cat(tokens, dim=0)  # [T_b, D]
                chain_tokens_list.append(tokens)
                chain_len_list.append(L)
                topo_dist_list.append(topo if topo is not None else None)
                block_valid_list.append(block_valid)
                continue

            # REPEAT tokens (random interleave per block)
            for blk in range(Bmax):
                seg_block_id_b = seg_block_id[b]  # Tensor [S]
                seg_dop_b = seg_dop[b]            # Tensor [S]
                
                seg_idx = [
                    s for s in range(S)
                    if int(seg_block_id_b[s]) == blk and int(seg_dop_b[s]) > 0
                ]
                if not seg_idx:
                    continue

                reps = []
                for s in seg_idx:
                    n = int(seg_dop[b][s])
                    reps.append(seg_repr[b][s].unsqueeze(0).repeat(n, 1))
                reps = torch.cat(reps, dim=0)

                perm = torch.randperm(reps.size(0), device=device)
                reps = reps[perm]

                tokens.append(reps)
                blocks.extend([blk] * reps.size(0))

            tokens = torch.cat(tokens, dim=0)   # [T_b, D]
            blocks = torch.tensor(blocks, device=device)
            # ===== Semantic Rewrite (junction-based) =====
            T_b = tokens.size(0)
            
            # ===== find junction index (last A repeat token) =====
            repeat_start = 2 + Bmax
            repeat_blocks = blocks[repeat_start:]
            
            # indices in full token space
            repeat_indices = torch.arange(repeat_start, T_b, device=device)
            
            # Only apply the legacy AB junction rewrite when there are exactly 2 blocks.
            if Bmax == 2:
                A_repeat_mask = repeat_blocks == 0
                if A_repeat_mask.any():
                    junction_idx = repeat_indices[A_repeat_mask].max()
                else:
                    junction_idx = repeat_start
            else:
                junction_idx = repeat_start
            
            # compute signed distance to junction
            idx = torch.arange(T_b, device=device)
            repeat_mask = blocks >= 0
            repeat_mask[:repeat_start] = False
            
            L_chain = repeat_mask.sum().float().clamp(min=1.0)

            dist = (idx - junction_idx).float() / (L_chain + 1e-6)
            dist = dist.unsqueeze(-1)
            
            # mask out non-repeat tokens (CLS/GLOB/BLOCK)
            rewrite_mask = torch.zeros(T_b, 1, device=device)
            rewrite_mask[repeat_start:] = 1.0   # only REPEAT tokens get rewritten
            
            # topological field
            topo_field = self.topo_mlp(dist) * rewrite_mask  # [T_b, D]
            
            # semantic rewrite (affine modulation)
            if Bmax == 2:
                tokens = tokens * (1.0 + topo_field) + self.topo_bias * rewrite_mask

            chain_tokens_list.append(tokens)
            chain_block_list.append(blocks)
            # legacy chain length counts repeat tokens only
            chain_len_list.append(int((blocks >= 0).sum().item()))
            topo_dist_list.append(None)
            block_valid_list.append(block_valid)

        # ---------- padding ----------
        max_T = max(t.size(0) for t in chain_tokens_list)

        chain_tokens = torch.zeros(B, max_T, D, device=device)
        chain_valid = torch.zeros(B, max_T, dtype=torch.bool, device=device)

        for b in range(B):
            T_b = chain_tokens_list[b].size(0)
            chain_tokens[b, :T_b] = chain_tokens_list[b]
            chain_valid[b, :T_b] = 1

        # ---------- attention mask ----------
        attn_mask = torch.full(
            (B, max_T, max_T),
            float("-inf"),
            device=device,
        )

        for b in range(B):
            valid = chain_valid[b]
            idx = torch.nonzero(valid).flatten()

            # allow all valid tokens to attend each other (then selectively block GLOB below)
            if idx.numel() > 0:
                attn_mask[b][idx[:, None], idx[None, :]] = 0.0

            # ensure diagonal is 0
            attn_mask[b].masked_fill_(torch.eye(max_T, device=device).bool(), 0.0)

            # If chain-level global feature is completely absent, block GLOB like stage-1.
            if glob_feat_mask is not None:
                mask_b = glob_feat_mask[b] if isinstance(glob_feat_mask, (list, tuple)) else glob_feat_mask
                if isinstance(mask_b, torch.Tensor):
                    glob_valid = float(mask_b.sum().item()) > 0.0
                else:
                    glob_valid = float(sum(mask_b)) > 0.0
                if not glob_valid:
                    attn_mask[b, 1, :] = float("-inf")
                    attn_mask[b, :, 1] = float("-inf")
                    attn_mask[b, 1, 1] = 0.0

            # Block invalid block tokens (pad them out like stage-1 SEG)
            if Bmax > 0 and b < len(block_valid_list):
                bv = block_valid_list[b]
                for blk in range(Bmax):
                    if not bool(bv[blk].item()):
                        t = 2 + blk
                        attn_mask[b, t, :] = float("-inf")
                        attn_mask[b, :, t] = float("-inf")
                        attn_mask[b, t, t] = 0.0

        attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

        # Pad topology distance matrices (repeat-token space) if provided in new mode.
        max_L = int(max(chain_len_list)) if chain_len_list else 0
        topo_padded = None
        if any(t is not None for t in topo_dist_list) and max_L > 0:
            topo_padded = torch.zeros(B, max_L, max_L, dtype=torch.long, device=device)
            for b in range(B):
                topo = topo_dist_list[b]
                L = int(chain_len_list[b])
                if topo is None or L == 0:
                    continue
                topo_padded[b, :L, :L] = topo

        meta = {
            "chain_len": torch.tensor(chain_len_list, dtype=torch.long, device=device),
            "repeat_start": 2 + Bmax,
            "topo_dist": topo_padded,
        }

        return chain_tokens, attn_mask, meta



class ChainEdgeFeaturePlus(nn.Module):

    def __init__(self, pair_dim, max_chain_dist):
        super().__init__()
        self.pair_dim = pair_dim
        self.max_chain_dist = max_chain_dist
        self.chain_dist_encoder = nn.Embedding(
             max_chain_dist + 1, pair_dim, padding_idx=0
        )
        self.vnode_virtual_distance = nn.Embedding(1, pair_dim)

    def forward(
        self,
        batched_data,
        graph_attn_bias,
        *,
        repeat_start: int = 4,
        chain_len: Optional[torch.Tensor] = None,
        topo_dist: Optional[torch.Tensor] = None,
    ):
        """Fill Stage-2 pair bias for repeat/node tokens.

        Args:
            repeat_start: index where repeat/node tokens begin in the stage-2 sequence.
            chain_len: Long[B] number of repeat/node tokens per sample.
            topo_dist: Optional Long[B, max_L, max_L] topology distances in repeat-token space.
                If None, falls back to linear |i-j| distance.
        """
        if chain_len is None:
            chain_len = batched_data["chain_len"]

        B = graph_attn_bias.size(0)
        device = graph_attn_bias.device

        for b in range(B):
            L = int(chain_len[b])
            if L <= 0:
                continue

            if topo_dist is not None:
                dist = topo_dist[b, :L, :L].to(device=device)
            else:
                idx = torch.arange(L, device=device)
                dist = torch.abs(idx[:, None] - idx[None, :])

            dist = torch.clamp(dist, max=self.max_chain_dist).long()
            graph_attn_bias[b,
                repeat_start:repeat_start + L,
                repeat_start:repeat_start + L,
                :
            ] = self.chain_dist_encoder(dist)

        t = self.vnode_virtual_distance.weight.view(1, 1, self.pair_dim)
        graph_attn_bias[:, :repeat_start, :, :] = t
        graph_attn_bias[:, :, :repeat_start, :] = t
        return graph_attn_bias

        
        
def build_chain_mask(*args, **kwargs):
    return None
        
######################Chain-level utils end####################

class SE3InvariantKernel(nn.Module):

    def __init__(
        self,
        pair_dim,
        num_pair,
        num_kernel,
        std_width=1.0,
        start=0.0,
        stop=9.0,
    ):
        super(SE3InvariantKernel, self).__init__()
        self.num_kernel = num_kernel

        self.gaussian = GaussianKernel(
            self.num_kernel,
            num_pair,
            std_width=std_width,
            start=start,
            stop=stop,
        )
        self.out_proj = NonLinear(self.num_kernel, pair_dim)

    def forward(self, dist, node_type_edge):
        edge_feature = self.gaussian(
            dist,
            node_type_edge.long(),
        )
        edge_feature = self.out_proj(edge_feature)
        return edge_feature


def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianKernel(nn.Module):
    def __init__(self, K=128, num_pair=512, std_width=1.0, start=0.0, stop=9.0):
        super().__init__()
        self.K = K
        std_width = std_width
        start = start
        stop = stop
        mean = torch.linspace(start, stop, K)
        self.std = (std_width * (mean[1] - mean[0]))
        self.register_buffer("mean", mean)
        self.mul = Embedding(num_pair, 1, padding_idx=0)
        self.bias = Embedding(num_pair, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1.0)

    def forward(self, x, atom_pair):
        mul = self.mul(atom_pair).abs().squeeze(-1)
        bias = self.bias(atom_pair).squeeze(-1)
        x = mul * x + bias
        x = x.unsqueeze(-1).expand(-1, -1, -1, self.K)
        mean = self.mean.float().view(-1)
        return gaussian(x.float(), mean, self.std)
        
        
class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = Linear(input, hidden, init="relu")
        self.layer2 = Linear(hidden, output_size, init="final")

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

    def zero_init(self):
        nn.init.zeros_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)


class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")


class DropPath(torch.nn.Module):
    def __init__(self, prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f"prob={self.drop_prob}"


class OuterProduct(nn.Module):
    def __init__(self, d_atom, d_pair, d_hid=32):
        super(OuterProduct, self).__init__()

        self.d_atom = d_atom
        self.d_pair = d_pair
        self.d_hid = d_hid

        self.linear_in = nn.Linear(d_atom, d_hid * 2)
        self.linear_out = nn.Linear(d_hid**2, d_pair)
        self.act = nn.GELU()
        self._memory_efficient = True

    def _opm(self, a, b):
        bsz, n, d = a.shape
        # outer = torch.einsum("...bc,...de->...bdce", a, b)
        a = a.view(bsz, n, 1, d, 1)
        b = b.view(bsz, 1, n, 1, d)
        outer = a * b
        outer = outer.view(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)
        return outer

    def forward(
        self,
        m: torch.Tensor,
        op_mask: Optional[torch.Tensor] = None,
        op_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        ab = self.linear_in(m) * op_mask
        a, b = ab.chunk(2, dim=-1)

        if self._memory_efficient and torch.is_grad_enabled():
            z = checkpoint(self._opm, a, b, use_reentrant=False)
        else:
            z = self._opm(a, b)

        z *= op_norm
        z = z * (op_mask @ op_mask.transpose(-2, -1)).unsqueeze(-1)
        return z


class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair, d_hid):
        super(TriangleMultiplication, self).__init__()

        self.linear_ab_p = Linear(d_pair, d_hid * 2)
        self.linear_ab_g = Linear(d_pair, d_hid * 2, init="gating")

        self.linear_g = Linear(d_pair, d_pair, init="gating")
        self.linear_z = Linear(d_hid, d_pair, init="final")

        self.layer_norm_out = nn.LayerNorm(d_hid)

    def _triangle_forward(self, z, mask, scale):
        mask = mask.unsqueeze(-1).float() * scale.unsqueeze(-1)
    
        g = self.linear_g(z)
        ab = self.linear_ab_p(z) * mask * torch.sigmoid(self.linear_ab_g(z))
        a, b = torch.chunk(ab, 2, dim=-1)
    
        a1 = permute_final_dims(a, (2, 0, 1))
        b1 = b.transpose(-1, -3)
        x = torch.matmul(a1, b1)
    
        b2 = permute_final_dims(b, (2, 0, 1))
        a2 = a.transpose(-1, -3)
        x = x + torch.matmul(a2, b2)
    
        x = permute_final_dims(x, (1, 2, 0))
    
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        return g * x
        
    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    
        if self.training and torch.is_grad_enabled():
            x = checkpoint(self._triangle_forward, z, mask, scale, use_reentrant=False)
        else:
            x = self._triangle_forward(z, mask, scale)
        return x


class Transition(nn.Module):
    def __init__(self, d_in, n, dropout=0.0):

        super(Transition, self).__init__()

        self.d_in = d_in
        self.n = n

        self.linear_1 = Linear(self.d_in, self.n * self.d_in, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(self.n * self.d_in, d_in, init="final")
        self.dropout = dropout

    def _transition(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_2(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self._transition(x=x)
        return x


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class MovementPredictionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        pair_dim: int,
        num_head: int,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.q_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.k_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.v_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.num_head = num_head
        self.scaling = (embed_dim // num_head) ** -0.5
        self.force_proj1 = Linear(embed_dim, 1, init="final", bias=False)
        self.linear_bias = Linear(pair_dim, num_head)
        self.pair_layer_norm = nn.LayerNorm(pair_dim)
        self.dropout = 0.1

    def zero_init(self):
        nn.init.zeros_(self.force_proj1.weight)

    def forward(
        self,
        query: Tensor,
        pair: Tensor,
        attn_mask: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        bsz, n_node, _ = query.size()
        query = self.layer_norm(query)
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_head, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        pair = self.pair_layer_norm(pair)
        bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()       
        attn = attn + bias
        attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        attn_probs = attn.view(bsz, self.num_head, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        cur_force = self.force_proj1(x).view(bsz, n_node, 3)
        return cur_force
        

class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.layer_norm(features)
        x = self.dense(x)
        x = self.act(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x