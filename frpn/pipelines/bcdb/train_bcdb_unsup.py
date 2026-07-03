#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dataloader.dataloader_polymer import build_dataloader
from main import apply_pkl_metadata_to_args, freeze_for_chain_stage1, load_monomer_pretrained_weights
from models.multi_mol_model import MultiMolModel


def parse_args():
    parser = argparse.ArgumentParser(description="BCDB unsupervised chain representation training")

    # Data and experiment identity
    parser.add_argument("--dataset_name", default="BCDB")
    parser.add_argument("--pkl_path", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--results_root", default="results_ssl_bcdb_unsup")

    # Model branch (fair comparison axis)
    parser.add_argument("--main_task", choices=["finetune", "chain", "stage2_only"], required=True)
    parser.add_argument("--stage2_only_repr", choices=["full", "smiles_only"], default="full")
    parser.add_argument("--disable_anchor_symmetry_break", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")

    # Optional initialization
    parser.add_argument("--weight_path", default="")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Validation and early-stop
    parser.add_argument("--val_every_steps", type=int, default=100)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument(
        "--early_stop_min_epochs",
        type=int,
        default=5,
        help="Do not trigger early-stop before this epoch (still tracks best checkpoint).",
    )
    parser.add_argument("--min_delta_abs", type=float, default=5e-4)
    parser.add_argument("--min_delta_rel", type=float, default=5e-3)
    parser.add_argument("--val_max_batches", type=int, default=64)

    # SSL objective configuration
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--rep_dropout", type=float, default=0.2)
    parser.add_argument("--w_contrast", type=float, default=1.0)
    parser.add_argument("--w_metric", type=float, default=1.0)
    parser.add_argument("--w_var", type=float, default=1.0)
    parser.add_argument("--w_cov", type=float, default=0.05)

    parser.add_argument(
        "--var_target_std",
        type=float,
        default=1.0,
        help="Target per-dimension std used by variance loss (applied on raw reps, not normalized reps).",
    )

    # Chemistry and topology definition
    parser.add_argument("--chem_weighting", choices=["presence", "dop", "balanced"], default="presence")
    parser.add_argument("--alpha_chem", type=float, default=0.5)
    parser.add_argument("--topo_weight_vf", type=float, default=1.0)
    parser.add_argument("--topo_weight_logdop", type=float, default=1.0)
    parser.add_argument("--topo_weight_junction", type=float, default=1.0)
    parser.add_argument("--max_metric_pairs", type=int, default=64)
    parser.add_argument("--mask_temperature_feature", action="store_true")

    # Distributed setup
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--dist-backend", dest="dist_backend", default="nccl")
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 1)))
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", 0)))

    # Model defaults compatible with existing code
    parser.add_argument("--equalize_active_params", action="store_true")
    parser.add_argument("--wo_geom_3d", action="store_true")
    parser.add_argument("--wo_triopm", action="store_true")
    parser.add_argument("--wo_edge", action="store_true")
    parser.add_argument("--wo_spd", action="store_true")
    parser.add_argument("--wo_pair", action="store_true")
    parser.add_argument("--wo_node", action="store_true")
    parser.add_argument("--wo_atom_feat", type=int, nargs="*", default=None)

    parser.add_argument("--num_chain_glob_feat", type=int, default=3)
    parser.add_argument("--num_chain_block_feat", type=int, default=1)
    parser.add_argument("--chain_pair_dim", type=int, default=32)
    parser.add_argument("--max_chain_dist", type=int, default=1536)
    parser.add_argument("--chain_encoder_layers", type=int, default=4)
    parser.add_argument("--chain_attention_heads", type=int, default=12)
    parser.add_argument("--chain_pair_hidden_dim", type=int, default=64)
    parser.add_argument("--chain_ffn_embed_dim", type=int, default=3072)
    parser.add_argument("--max_chain_tokens", type=int, default=1024)
    parser.add_argument("--max_segments", type=int, default=10)
    parser.add_argument("--num_block_types", type=int, default=8)
    parser.add_argument("--num_seg_smiles_types", type=int, default=20000)

    parser.add_argument("--task_type", choices=["reg", "cls"], default="reg")
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--encoder_embed_dim", type=int, default=768)
    parser.add_argument("--padding_idx", type=int, default=0)
    parser.add_argument("--num_atom", type=int, default=512)
    parser.add_argument("--num_degree", type=int, default=128)
    parser.add_argument("--pair_embed_dim", type=int, default=512)
    parser.add_argument("--num_edge", type=int, default=64)
    parser.add_argument("--num_spatial", type=int, default=512)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--pair_hidden_dim", type=int, default=64)
    parser.add_argument("--encoder_ffn_embed_dim", type=int, default=768)
    parser.add_argument("--encoder_attention_heads", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--activation_dropout", type=float, default=0.1)
    parser.add_argument("--activation_fn", type=str, default="gelu")
    parser.add_argument("--droppath_prob", type=float, default=0.0)
    parser.add_argument("--pair_dropout", type=float, default=0.1)
    parser.add_argument("--num_pair", type=int, default=512)
    parser.add_argument("--num_kernel", type=int, default=128)
    parser.add_argument("--gaussian_std_width", type=float, default=1.0)
    parser.add_argument("--gaussian_mean_start", type=float, default=0.0)
    parser.add_argument("--gaussian_mean_stop", type=float, default=9.0)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(args):
    args.distributed = args.distributed or args.world_size > 1
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.local_rank)
    else:
        args.rank = 0
        args.world_size = 1


def is_main_process(args) -> bool:
    return args.rank == 0


def maybe_mask_temperature_feature(batch, enabled: bool):
    if not enabled:
        return

    if "glob_feat" in batch and "glob_mask" in batch:
        glob_feat = batch["glob_feat"]
        glob_mask = batch["glob_mask"]
        if torch.is_tensor(glob_feat) and torch.is_tensor(glob_mask) and glob_feat.dim() == 2 and glob_feat.size(1) > 0:
            glob_feat[:, 0] = 0.0
            glob_mask[:, 0] = 0.0
            if "glob_valid_mask" in batch and torch.is_tensor(batch["glob_valid_mask"]):
                batch["glob_valid_mask"] = (glob_mask.sum(dim=1, keepdim=True) > 0).float()

    if "chain_glob_feat" in batch and "chain_glob_mask" in batch:
        c_feat = batch["chain_glob_feat"]
        c_mask = batch["chain_glob_mask"]
        if torch.is_tensor(c_feat) and torch.is_tensor(c_mask) and c_feat.dim() == 2 and c_feat.size(1) > 0:
            c_feat[:, 0] = 0.0
            c_mask[:, 0] = 0.0


def gather_with_local_grad(x: torch.Tensor, args) -> torch.Tensor:
    if not args.distributed or not dist.is_initialized():
        return x

    x = x.contiguous()

    world = dist.get_world_size()
    rank = dist.get_rank()
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x)
    xs[rank] = x
    return torch.cat(xs, dim=0)


def gather_no_grad(x: torch.Tensor, args) -> torch.Tensor:
    if not args.distributed or not dist.is_initialized():
        return x

    x = x.contiguous()

    world = dist.get_world_size()
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x)
    return torch.cat(xs, dim=0)


def _chem_signature(smiles_ids: np.ndarray, seg_valid: np.ndarray, seg_dop: np.ndarray, weighting: str) -> Dict[int, float]:
    valid = (seg_valid > 0.5) & (seg_dop > 0)
    ids = smiles_ids[valid].astype(np.int64)
    dops = seg_dop[valid].astype(np.float64)

    if ids.size == 0:
        return {0: 1.0}

    if weighting == "presence":
        uniq = np.unique(ids)
        w = 1.0 / float(len(uniq))
        return {int(k): w for k in uniq.tolist()}

    dop_map: Dict[int, float] = {}
    dop_sum = float(max(dops.sum(), 1e-8))
    for sid, dop in zip(ids.tolist(), dops.tolist()):
        dop_map[int(sid)] = dop_map.get(int(sid), 0.0) + float(dop) / dop_sum

    if weighting == "dop":
        return dop_map

    uniq = np.unique(ids)
    p_map = {int(k): 1.0 / float(len(uniq)) for k in uniq.tolist()}
    keys = set(dop_map.keys()) | set(p_map.keys())
    out: Dict[int, float] = {}
    for k in keys:
        out[k] = 0.5 * dop_map.get(k, 0.0) + 0.5 * p_map.get(k, 0.0)
    return out


def _sparse_cosine_distance(a: Dict[int, float], b: Dict[int, float]) -> float:
    keys = set(a.keys()) | set(b.keys())
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k in keys:
        va = a.get(k, 0.0)
        vb = b.get(k, 0.0)
        dot += va * vb
        na += va * va
        nb += vb * vb
    denom = math.sqrt(max(na, 1e-12) * max(nb, 1e-12))
    return 1.0 - dot / denom


def _junction_sym(seg_dop: np.ndarray, seg_block_id: np.ndarray, seg_valid: np.ndarray) -> float:
    valid = (seg_valid > 0.5) & (seg_dop > 0)
    dop = seg_dop[valid].astype(np.float64)
    blk = seg_block_id[valid].astype(np.int64)
    if dop.size == 0:
        return 0.5
    b0 = dop[blk == 0].sum()
    b1 = dop[blk == 1].sum()
    denom = max(float(b0 + b1), 1e-8)
    j = float(b0 / denom)
    return min(j, 1.0 - j)


def _vf_sym(block_feat: np.ndarray, block_feat_mask: np.ndarray) -> float:
    has0 = block_feat_mask[0, 0] > 0.5
    has1 = block_feat_mask[1, 0] > 0.5
    if has0 and has1:
        f1 = float(block_feat[0, 0])
        f2 = float(block_feat[1, 0])
        return min(f1, f2)
    if has0:
        f1 = float(block_feat[0, 0])
        return min(f1, 1.0 - f1)
    if has1:
        f2 = float(block_feat[1, 0])
        return min(f2, 1.0 - f2)
    return 0.5


def build_target_distance(
    seg_smiles_id: torch.Tensor,
    seg_valid: torch.Tensor,
    seg_dop: torch.Tensor,
    seg_block: torch.Tensor,
    block_feat: torch.Tensor,
    block_feat_mask: torch.Tensor,
    args,
) -> torch.Tensor:
    smiles_np = seg_smiles_id.detach().cpu().numpy()
    valid_np = seg_valid.detach().cpu().numpy()
    dop_np = seg_dop.detach().cpu().numpy()
    block_np = seg_block.detach().cpu().numpy()
    bfeat_np = block_feat.detach().cpu().numpy()
    bmask_np = block_feat_mask.detach().cpu().numpy()

    n = smiles_np.shape[0]
    chem_sig: List[Dict[int, float]] = []
    topo = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        chem_sig.append(_chem_signature(smiles_np[i], valid_np[i], dop_np[i], args.chem_weighting))
        topo[i, 0] = _vf_sym(bfeat_np[i], bmask_np[i])
        topo[i, 1] = math.log(max(float(dop_np[i].sum()), 1.0))
        topo[i, 2] = _junction_sym(dop_np[i], block_np[i], valid_np[i])

    topo_scales = np.median(np.abs(topo - np.median(topo, axis=0, keepdims=True)), axis=0)
    topo_scales = np.clip(topo_scales, 1e-6, None)

    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d_chem = _sparse_cosine_distance(chem_sig[i], chem_sig[j])
            d_topo = (
                args.topo_weight_vf * abs(topo[i, 0] - topo[j, 0]) / topo_scales[0]
                + args.topo_weight_logdop * abs(topo[i, 1] - topo[j, 1]) / topo_scales[1]
                + args.topo_weight_junction * abs(topo[i, 2] - topo[j, 2]) / topo_scales[2]
            )
            d = args.alpha_chem * d_chem + (1.0 - args.alpha_chem) * d_topo
            out[i, j] = d
            out[j, i] = d

    return torch.tensor(out, dtype=torch.float32, device=seg_smiles_id.device)


def sample_pair_indices(n: int, max_pairs: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if n < 2:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

    i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=device)
    num_pairs = i_idx.numel()
    if num_pairs <= max_pairs:
        return i_idx, j_idx

    perm = torch.randperm(num_pairs, device=device)[:max_pairs]
    return i_idx[perm], j_idx[perm]


def metric_distance_loss(
    z: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    args,
    gather_world: bool,
) -> torch.Tensor:
    if gather_world:
        z_use = gather_with_local_grad(z, args)
        seg_smiles_id = gather_no_grad(batch["seg_smiles_id"], args)
        seg_valid = gather_no_grad(batch["seg_valid_mask"], args)
        seg_dop = gather_no_grad(batch["seg_dop"], args)
        seg_block = gather_no_grad(batch["seg_block_id"], args)
        block_feat = gather_no_grad(batch["block_feat"], args)
        block_feat_mask = gather_no_grad(batch["block_feat_mask"], args)
    else:
        z_use = z
        seg_smiles_id = batch["seg_smiles_id"]
        seg_valid = batch["seg_valid_mask"]
        seg_dop = batch["seg_dop"]
        seg_block = batch["seg_block_id"]
        block_feat = batch["block_feat"]
        block_feat_mask = batch["block_feat_mask"]

    n = z_use.size(0)
    if n < 2:
        return z_use.new_tensor(0.0)

    target_mat = build_target_distance(seg_smiles_id, seg_valid, seg_dop, seg_block, block_feat, block_feat_mask, args)
    i_idx, j_idx = sample_pair_indices(n, args.max_metric_pairs, z_use.device)
    if i_idx.numel() == 0:
        return z_use.new_tensor(0.0)

    latent_dist = 1.0 - (z_use[i_idx] * z_use[j_idx]).sum(dim=-1)
    target_dist = target_mat[i_idx, j_idx]

    target_scale = target_dist.detach().mean().clamp_min(1e-6)
    target_dist = target_dist / target_scale

    return F.smooth_l1_loss(latent_dist, target_dist)
def variance_loss(z: torch.Tensor, target_std: float) -> torch.Tensor:
    if z.size(0) < 2:
        return z.new_tensor(0.0)
    z = z.float()
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
    return F.relu(float(target_std) - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.size(0) < 2:
        return z.new_tensor(0.0)
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.t() @ z) / (z.size(0) - 1)
    d = cov.size(0)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / max(d, 1)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, args) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    z1_all = gather_with_local_grad(z1, args)
    z2_all = gather_with_local_grad(z2, args)

    bs = z1.size(0)
    if bs == 0:
        return z1.new_tensor(0.0)

    if args.distributed and dist.is_initialized():
        rank = dist.get_rank()
        global_labels = torch.arange(bs, device=z1.device) + rank * bs
    else:
        global_labels = torch.arange(bs, device=z1.device)

    logits_12 = (z1 @ z2_all.t()) / args.temperature
    logits_21 = (z2 @ z1_all.t()) / args.temperature
    loss_12 = F.cross_entropy(logits_12, global_labels)
    loss_21 = F.cross_entropy(logits_21, global_labels)
    return 0.5 * (loss_12 + loss_21)


class RepHook:
    def __init__(self, module: torch.nn.Module):
        self.rep = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.rep = inputs[0]

    def pop(self) -> torch.Tensor:
        if self.rep is None:
            raise RuntimeError("Representation hook did not capture reg_head input.")
        out = self.rep
        self.rep = None
        return out

    def close(self):
        self.handle.remove()


def run_model_and_get_rep(model, batch, rep_hook: RepHook) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = model(batch)
    rep = rep_hook.pop()
    return rep, pred


def evaluate_unsup(model, val_loader, rep_hook: RepHook, args) -> Dict[str, float]:
    model.eval()

    sum_total = 0.0
    sum_metric = 0.0
    sum_var = 0.0
    sum_cov = 0.0
    count = 0

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if args.val_max_batches > 0 and bi >= args.val_max_batches:
                break

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda(non_blocking=True)
            maybe_mask_temperature_feature(batch, args.mask_temperature_feature)

            rep, _ = run_model_and_get_rep(model, batch, rep_hook)
            z_norm = F.normalize(rep, dim=-1)
            l_metric = metric_distance_loss(z_norm, batch, args, gather_world=False)

            # Anti-collapse should be computed on raw representations.
            rep_for_stats = rep
            if args.distributed and dist.is_initialized():
                rep_for_stats = gather_no_grad(rep_for_stats, args)
            l_var = variance_loss(rep_for_stats, target_std=args.var_target_std)
            l_cov = covariance_loss(rep_for_stats)
            total = args.w_metric * l_metric + args.w_var * l_var + args.w_cov * l_cov

            sum_total += float(total.item())
            sum_metric += float(l_metric.item())
            sum_var += float(l_var.item())
            sum_cov += float(l_cov.item())
            count += 1

    if args.distributed and dist.is_initialized():
        t = torch.tensor([sum_total, sum_metric, sum_var, sum_cov, count], dtype=torch.float32, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sum_total, sum_metric, sum_var, sum_cov, count = t.tolist()

    denom = max(int(count), 1)
    return {
        "val_total": sum_total / denom,
        "val_metric": sum_metric / denom,
        "val_var": sum_var / denom,
        "val_cov": sum_cov / denom,
    }


def build_run_dir(args) -> Path:
    run_dir = (
        Path(args.results_root)
        / args.dataset_name
        / f"unsup_task_{args.main_task}_chem_{args.chem_weighting}_maskT_{int(args.mask_temperature_feature)}"
    )

    if args.main_task in {"chain", "stage2_only"}:
        run_dir = run_dir / f"repr_{args.stage2_only_repr}_noanchor_{int(args.disable_anchor_symmetry_break)}"

    run_dir = run_dir / f"fold_{args.fold}" / f"seed_{args.seed}_bs_{args.batch_size}_lr_{args.lr}_wd_{args.weight_decay}"
    return run_dir


def main():
    args = parse_args()
    init_distributed(args)
    set_seed(args.seed + args.rank)

    apply_pkl_metadata_to_args(args)

    model = MultiMolModel(args)

    if args.weight_path:
        if Path(args.weight_path).is_file():
            model = load_monomer_pretrained_weights(model, args.weight_path, args)
        elif is_main_process(args):
            print(f"[WARN] weight_path not found, train from scratch: {args.weight_path}")

    if args.main_task == "chain" and args.freeze_encoder:
        freeze_for_chain_stage1(model)

    model = model.cuda()
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    reg_head_module = model.module.reg_head if isinstance(model, DDP) else model.reg_head
    rep_hook = RepHook(reg_head_module)

    train_loader, train_sampler = build_dataloader(args.pkl_path, args, mode="train")
    val_args = argparse.Namespace(**vars(args))
    val_args.batch_size = max(int(args.val_batch_size), 1)
    val_loader, _ = build_dataloader(args.pkl_path, val_args, mode="val")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    scaler = GradScaler(enabled=args.amp)

    run_dir = build_run_dir(args)
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    if is_main_process(args):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, Dict[str, float]] = {}
    best_val = None
    best_step = 0
    intervals_no_improve = 0
    global_step = 0

    stop_training = False

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        pbar = tqdm(train_loader, disable=not is_main_process(args), desc=f"Unsup-E{epoch}")

        for batch in pbar:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda(non_blocking=True)
            maybe_mask_temperature_feature(batch, args.mask_temperature_feature)

            with autocast(enabled=args.amp):
                rep, pred = run_model_and_get_rep(model, batch, rep_hook)

                # Build two stochastic views in representation space to avoid
                # DDP multi-forward graph/version conflicts.
                view1 = F.dropout(rep, p=args.rep_dropout, training=True)
                view2 = F.dropout(rep, p=args.rep_dropout, training=True)
                z1 = F.normalize(view1, dim=-1)
                z2 = F.normalize(view2, dim=-1)
                z_mid = F.normalize(0.5 * (z1 + z2), dim=-1)

                l_contrast = contrastive_loss(z1, z2, args)
                l_metric = metric_distance_loss(z_mid, batch, args, gather_world=True)

                # Anti-collapse is computed on raw (unnormalized) reps; gather across DDP
                # to reduce small-batch noise.
                rep_cat = torch.cat([view1, view2], dim=0)
                if args.distributed and dist.is_initialized():
                    rep_cat = gather_with_local_grad(rep_cat, args)
                l_var = variance_loss(rep_cat, target_std=args.var_target_std)
                l_cov = covariance_loss(rep_cat)

                loss = (
                    args.w_contrast * l_contrast
                    + args.w_metric * l_metric
                    + args.w_var * l_var
                    + args.w_cov * l_cov
                ) / args.grad_accum_steps

                # Keep DDP graph tracing aware that forward outputs are used.
                loss = loss + pred.float().sum() * 0.0

            scaler.scale(loss).backward()

            if (global_step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1

            if is_main_process(args) and global_step % 20 == 0:
                pbar.set_postfix(
                    loss=float(loss.item() * args.grad_accum_steps),
                    l_contrast=float(l_contrast.item()),
                    l_metric=float(l_metric.item()),
                    l_var=float(l_var.item()),
                    l_cov=float(l_cov.item()),
                )

            if global_step % args.val_every_steps == 0:
                val_metrics = evaluate_unsup(model, val_loader, rep_hook, args)
                cur_val = float(val_metrics["val_total"])

                if is_main_process(args):
                    print(
                        f"[Step {global_step}] val_total={cur_val:.5f} "
                        f"val_metric={val_metrics['val_metric']:.5f} "
                        f"val_var={val_metrics['val_var']:.5f} "
                        f"val_cov={val_metrics['val_cov']:.5f}"
                    )

                    if best_val is None:
                        improve = True
                    else:
                        improve = (best_val - cur_val) > max(args.min_delta_abs, best_val * args.min_delta_rel)

                    if improve:
                        best_val = cur_val
                        best_step = global_step
                        intervals_no_improve = 0

                        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                        torch.save(
                            {
                                "global_step": global_step,
                                "epoch": epoch,
                                "model_state": model_state,
                                "optimizer_state": optimizer.state_dict(),
                                "scaler_state": scaler.state_dict(),
                                "scheduler_state": scheduler.state_dict(),
                                "best_val": best_val,
                                "best_step": best_step,
                                "intervals_no_improve": intervals_no_improve,
                                "args": vars(args),
                            },
                            ckpt_dir / "best.pt",
                        )
                    else:
                        intervals_no_improve += 1

                    history[str(global_step)] = {
                        "epoch": epoch,
                        "val_total": cur_val,
                        "val_metric": float(val_metrics["val_metric"]),
                        "val_var": float(val_metrics["val_var"]),
                        "val_cov": float(val_metrics["val_cov"]),
                        "best_val": float(best_val) if best_val is not None else None,
                        "best_step": int(best_step),
                        "intervals_no_improve": int(intervals_no_improve),
                    }
                    with (log_dir / "metrics.json").open("w") as f:
                        json.dump(history, f, indent=2)

                    stop_flag = (epoch >= int(args.early_stop_min_epochs)) and (
                        intervals_no_improve >= args.early_stop_patience
                    )
                    flag_tensor = torch.tensor(int(stop_flag), device="cuda")
                else:
                    flag_tensor = torch.zeros(1, device="cuda")

                if args.distributed and dist.is_initialized():
                    dist.broadcast(flag_tensor, src=0)
                stop_training = bool(flag_tensor.item())

                model.train()
                if stop_training:
                    if is_main_process(args):
                        print(
                            f"[Early-Stop] stop at step={global_step}, best_step={best_step}, best_val={best_val:.5f}"
                        )
                    break

        scheduler.step()
        if stop_training:
            break

    rep_hook.close()

    if is_main_process(args):
        with (run_dir / "RUN_INFO.json").open("w") as f:
            json.dump(
                {
                    "run_dir": str(run_dir),
                    "best_checkpoint": str(ckpt_dir / "best.pt"),
                    "best_val": best_val,
                    "best_step": best_step,
                    "stopped_early": bool(stop_training),
                },
                f,
                indent=2,
            )
        print(f"[DONE] run_dir={run_dir}")

    if args.distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
