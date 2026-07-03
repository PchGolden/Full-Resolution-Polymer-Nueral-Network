# -*- coding: utf-8 -*-
from dataloader_polymer import PolymerPickleDataset
from torch.utils.data import DataLoader
import pickle, torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
from argparse import Namespace

#
def train_one_epoch(model, loader, optimizer, scaler, epoch, args, sampler):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    running_loss = 0.0
    total_samples = 0
    
    for step, batch in tqdm(enumerate(loader), total=len(loader), disable=not is_main_process(args.rank), desc=f"Epoch {epoch}"):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)

        with autocast(enabled=args.amp):
            pred = model(batch)  # [B, num_tasks]
            label = batch["label"].cuda()
            if args.task_type == 'reg':
                loss = torch.nn.functional.mse_loss(pred, label) / args.grad_accum_steps
            else:
                loss = torch.nn.functional.cross_entropy(pred, label.long()) / args.grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum_steps == 0 or (step + 1 == len(loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * label.size(0) * args.grad_accum_steps
        total_samples += label.size(0)

    if args.distributed:
        tensor_loss = torch.tensor(running_loss, device="cuda")
        dist.all_reduce(tensor_loss, op=dist.ReduceOp.SUM)
        running_loss = tensor_loss.item()
        
        tensor_samples = torch.tensor(total_samples, device="cuda")
        dist.all_reduce(tensor_samples, op=dist.ReduceOp.SUM)
        total_samples = tensor_samples.item()

    epoch_loss = running_loss / total_samples
    return epoch_loss
#

# same PAD constants as UniMol2
PAD_TOKEN_ID   = 0      # <- pad for src_token
PAD_FEAT_VAL   = 0      # <- pad for discrete features
PAD_SPD_VAL    = 511    # <- unreachable shortest-path  (already +1)
PAD_BIAS_VAL   = float('-inf')


dataset = PolymerPickleDataset("/share/home/wujt/diblock/unimol2/data/test/Tg.pkl")

# ---------- dataloader_polymer.py ----------
import pickle
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

# same PAD constants as UniMol2
PAD_TOKEN_ID = 0         # pad for src_token
PAD_FEAT_VAL = 0         # pad for discrete features
PAD_SPD_VAL = 0        # unreachable shortest-path (already +1)
PAD_BIAS_VAL = float("-inf")  # pad for attention bias

# ---------- 1. Dataset -------------------------------------------------
class PolymerPickleDataset(Dataset):
    """
    A thin loader around the *.pkl produced by preprocessing_polymer.py
    """
    def __init__(self, pkl_path: str, split: str | None = None) -> None:
        header = pickle.load(open(pkl_path, "rb"))
        self.samples: List[Dict[str, Any]] = [
            s for s in header["samples"]
            if split is None or s["split"] == split
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------- 2. Padding helpers ----------------------------------------
def pad_1d(samples: List[torch.Tensor], pad_len: int, pad_value=0):
    batch_size = len(samples)
    tensor = samples[0].new_full((batch_size, pad_len), pad_value)
    for i, x in enumerate(samples):
        tensor[i, : x.shape[0]] = x
    return tensor

def pad_1d_feat(samples: List[torch.Tensor], pad_len: int, pad_value=0):
    batch_size = len(samples)
    feat_size = samples[0].shape[-1]
    tensor = samples[0].new_full((batch_size, pad_len, feat_size), pad_value)
    for i, x in enumerate(samples):
        tensor[i, : x.shape[0]] = x
    return tensor

def pad_2d(samples: List[torch.Tensor], pad_len: int, pad_value=0):
    batch_size = len(samples)
    tensor = samples[0].new_full((batch_size, pad_len, pad_len), pad_value)
    for i, x in enumerate(samples):
        n, m = x.shape
        tensor[i, :n, :m] = x
    return tensor

def pad_2d_feat(samples: List[torch.Tensor], pad_len: int, pad_value=0):
    batch_size = len(samples)
    feat_size = samples[0].shape[-1]
    tensor = samples[0].new_full((batch_size, pad_len, pad_len, feat_size), pad_value)
    for i, x in enumerate(samples):
        n, m, _ = x.shape
        tensor[i, :n, :m] = x
    return tensor

def pad_base_mask(samples: List[torch.Tensor], pad_len: int):
    batch_size = len(samples)
    # note: official adds +1 inside
    tensor = samples[0].new_full((batch_size, pad_len, pad_len),
                                 float("-inf"))
    for i, b in enumerate(samples):
        n, m = b.shape
        # copy real block
        tensor[i, :n, :m] = b
        # allow padded *rows* to attend to real *cols*
        tensor[i, n:, :m] = 0
    return tensor


# ---------- 3. collate_fn ---------------------------------------------
def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 0) Ensure each sample has atom_mask
    for s in samples:
        N = s["atom_feat"].shape[0]
        s["atom_mask"] = torch.ones(N, dtype=torch.long)

    # ---------- 1. Calculate padding length ----------
    special_T   = samples[0]['special_T']
    max_node    = max(s["atom_mask"].shape[0] for s in samples)
    token_dim   = (max_node + special_T + 3) // 4 * 4  # Multiple of 4: e.g. 12
    node_pad_len = token_dim - special_T               # e.g. 8

    # ---------- 2. Define padding functions ----------
    pad_fns = {
        "src_token"     : (pad_1d,       PAD_TOKEN_ID),
        "src_pos"       : (pad_1d_feat,  0.0),
        "atom_feat"     : (pad_1d_feat,  PAD_FEAT_VAL),
        "atom_mask"     : (pad_1d,       0),
        "edge_feat"     : (pad_2d_feat,  PAD_FEAT_VAL),
        "shortest_path" : (pad_2d,       PAD_SPD_VAL),
        "degree"        : (pad_1d,       PAD_FEAT_VAL),
        "pair_type"     : (pad_2d_feat,  PAD_FEAT_VAL),
        "segment_id"    : (pad_1d,       -1),          # Can also be listed separately
    }

    batched: Dict[str, Any] = {}

    for key in samples[0].keys():
        vals = [s[key] for s in samples]

        # ---------- 3a. Keys that need padding based on atom count ----------
        if key in pad_fns:
            fn, pad_val = pad_fns[key]
            pad_len = node_pad_len
            batched[key] = fn(vals, pad_len, pad_val)

        # ---------- 3b. Fixed-length keys, directly stack ----------
        elif key in {
            "glob_feat", "glob_mask", "glob_valid_mask",
            "seg_feat", "seg_feat_mask", "seg_valid_mask",
        }:
            batched[key] = torch.stack(vals)

        # ---------- 3c. Label ----------
        elif key == "label":
            batched[key] = torch.stack([
                torch.tensor(list(v.values()), dtype=torch.float32) for v in vals
            ])
            
        # ---------- 3e. base_mask ----------
        elif key == "base_mask":                # The only exception
            fn = pad_base_mask
            pad_len = token_dim
            batched[key] = fn(vals, pad_len)

        # ---------- 3d. Keep the rest as a list ----------
        else:
            batched[key] = vals
        

    return batched
# ----------------------------------------------------------------------


dataloader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=collate_fn,
    shuffle=False 
)

batch = next(iter(dataloader))
print(batch)

import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

args = Namespace(
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_ffn_embed_dim=768,
    encoder_attention_heads=48,
    activation_fn="relu",
    pooler_activation_fn="tanh",
    emb_dropout=0.1,
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.1,
    pooler_dropout=0.1,
    max_seq_len=512,
    post_ln=True,
    masked_token_loss=1.0,
    masked_dist_loss=0.1,
    masked_coord_loss=0.1,
    masked_coord_dist_loss=0.1,
    pair_embed_dim=512,
    pair_hidden_dim=64,
    pair_dropout=0.1,
    droppath_prob=0.1,
    notri=False,
    gaussian_std_width=1.0,
    gaussian_mean_start=0.0,
    gaussian_mean_stop=9.0,
    mode="train",
    num_atom = 512,
    num_degree = 128,
    num_edge = 64,
    num_pair = 512,
    num_spatial = 512,
)

from multi_mol_model import MultiMolModel
from utils import _build_attn_mask
model = MultiMolModel(args)
model.load_state_dict(torch.load('./checkpoint.pt')['model'], strict=False)
with torch.no_grad():
    model.eval()
    # ---------- 0. Unpack inputs ----------
    base_mask  = batch["base_mask"]        # [B, T, T] - includes CLS + GLOB + SEG + ATOM
    atom_mask  = batch["atom_mask"]        # [B, N_atom] - (1/0, padding = 0)
    seg_id     = batch["segment_id"]       # [B, N_atom]
    pos        = batch["src_pos"]          # [B, N_atom, 3]
    pair_type  = batch["pair_type"]        # [B, N_atom, N_atom]
    token_feat = model.embed_tokens(batch["src_token"])
    
    B, N_atom  = atom_mask.shape
    n_seg      = batch["seg_feat"].size(1)
    T          = 1 + 1 + n_seg + N_atom            # Total tokens: CLS + GLOB + SEG + ATOM
    special_len = 2 + n_seg                        # CLS + GLOB + SEG
    device     = atom_mask.device
    
    # ---------- 1. Node features ----------
    x = model.atom_feature(batch, token_feat).type(torch.float32)  # [B, T, D]
    
    # ---------- 2. Attention mask (-inf / 0) ----------
    attn_mask = _build_attn_mask(
        batch["base_mask"], batch["segment_id"], atom_mask, batch["seg_valid_mask"]
    )
    
    # ---------- 3. Graph attention bias ----------
    graph_attn_bias = x.new_zeros(B, T, T, args.pair_embed_dim)
    graph_attn_bias = model.edge_feature(batch, graph_attn_bias)
    
    # ---------- 4. 3D bias (applied only within same segment) ----------
    delta_pos     = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
    dist          = delta_pos.norm(dim=-1)               # [B, N, N]
    attn_bias_3d  = model.se3_invariant_kernel(dist.detach(), pair_type.long())  # [B, N, N, Dp]
    
    same_seg_atoms = (seg_id.unsqueeze(-1) == seg_id.unsqueeze(-2))  # [B, N, N]
    attn_bias_3d.masked_fill_(~same_seg_atoms.unsqueeze(-1), 0.)     # mask inter-segment pairs
    
    # Insert into graph_attn_bias in atom-atom block
    graph_attn_bias[:, special_len:, special_len:, :].add_(attn_bias_3d)
    
    # ---------- 5. Project attention bias to attention heads ----------
    #attn_bias = self.attn_bias_proj(graph_attn_bias)      # [B, T, T, H]
    #attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous() # [B, H, T, T]
    #Note: Save this projection to Attention's forward
    
    # ---------- 6. Padding masks ----------
    cls_mask  = torch.ones(B, 1, device=device, dtype=torch.bool)
    glob_valid = batch["glob_valid_mask"].bool()    # [B,1]
    seg_valid  = batch["seg_valid_mask"].bool()     # [B,S]
    node_mask = torch.cat([cls_mask, glob_valid, seg_valid, atom_mask.bool()], dim=1)  # [B, T]
    
    # Pair mask: only valid token pairs
    pair_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)  # [B, T, T]
    # ---------- 7. Run encoder ----------
    x, pair = model.encoder(
        x,                 # [B, T, D]
        graph_attn_bias,         # [B, H, T, T]
        atom_mask=node_mask,  # [B, T]
        pair_mask=pair_mask,  # [B, T, T]
        attn_mask=attn_mask,  # [B, H, T, T]
    )
    