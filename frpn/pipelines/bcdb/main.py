#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aligned with directory structure and model/dataloader definitions.

Usage Example (single node with 4 GPUs):
    torchrun --standalone --nproc_per_node 4 frpn/pipelines/bcdb/main.py \
        --dataset_name polymer1 --fold 0 \
        --pkl_path data/processed/train_processed.pkl \
        --epochs 100 --lr 5e-4 --weight_decay 1e-4
"""

import argparse, json, math, os, random, time, warnings
import pickle
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
# ---------- Internal Modules -----------------------
from models.multi_mol_model import MultiMolModel
from dataloader.dataloader_polymer import build_dataloader
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

warnings.filterwarnings("ignore", category=UserWarning)

METRIC_PRINT_DIGITS = 8


# ==================================================
#  Entry Point
# ==================================================
def parse_args():
    parser = argparse.ArgumentParser(description="DDP Trainer")
    
    # ---- Ablation Studys -------------------------
    parser.add_argument("--wo_geom_3d", action="store_true")
    parser.add_argument("--wo_triopm", action="store_true")
    parser.add_argument("--wo_edge", action="store_true")
    parser.add_argument("--wo_spd", action="store_true")
    parser.add_argument("--wo_pair", action="store_true")
    parser.add_argument("--wo_node", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--wo_atom_feat", type=int, nargs="*", default=None)
    parser.add_argument("--equalize_active_params", action="store_true")

    # ---- Dataset / Path -----------------------------------
    parser.add_argument("--dataset_name",help="Name of the dataset")
    parser.add_argument("--non_kfold", default=False, action="store_true", help="Use separate train/test pkl instead of CV folds")
    parser.add_argument("--pkl_path", help="Path to the *.pkl file")
    parser.add_argument("--test_pkl_path", help="Path to the *_test.pkl file")
    parser.add_argument("--weight_path", default="/share/home/202320162823/unimacro/no_pretrain.pt")
    parser.add_argument("--pretrain_train_path", default="/public/home/202320162823/version2_pretrain/pretrain_data/pretrain_train.lmdb")
    parser.add_argument("--pretrain_val_path",default="/public/home/202320162823/version2_pretrain/pretrain_data/pretrain_val.lmdb")
    parser.add_argument("--fold", type=int, help="Cross-validation fold index")
    parser.add_argument("--results_root", default="results", help="Root directory to save results")
    parser.add_argument(
        "--save_finetune_checkpoint",
        action="store_true",
        help="If set, save best checkpoint.pt for finetune runs as well.",
    )

    # ---- Training Hyperparameters -------------------------
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--early_stop_patience", type=int, default=200)
    parser.add_argument("--val_every_steps", type=int, default=50)
    parser.add_argument("--min_delta_abs", type=float, default=1e-3)
    parser.add_argument("--min_delta_rel", type=float, default=1e-2)
    parser.add_argument(
        "--label_zscore",
        action="store_true",
        help=(
            "Regression only: apply per-fold target transform + z-score using train split stats, "
            "train in normalized space, and report both z/raw metrics."
        ),
    )
    parser.add_argument(
        "--log10_labels",
        default="",
        help="Comma-separated label names to apply log10 transform before z-score (e.g. 'D,self-diffusion').",
    )
    parser.add_argument(
        "--log10_clamp_min",
        type=float,
        default=1e-16,
        help="Clamp minimum for log10 labels (values <=0 will be clamped).",
    )
    parser.add_argument(
        "--pretrain_use_split",
        action="store_true",
        help="Use fold split (train/val) for pretrain data instead of full/full loaders.",
    )
    parser.add_argument(
        "--pretrain_split_fold",
        type=int,
        default=0,
        help="Fold index used when --pretrain_use_split is enabled.",
    )
    parser.add_argument(
        "--mask_temperature_feature",
        action="store_true",
        help="Mask temperature feature (glob_feat0 / chain_glob_feat[0]) to avoid non-chemical/topology noise.",
    )
    
    # ---- Chain-level model args ----
    parser.add_argument("--num_chain_glob_feat", type=int, default=3)   # T, Mn, coexistence
    parser.add_argument("--num_chain_block_feat", type=int, default=1)  # volume fraction
    
    parser.add_argument("--chain_pair_dim", type=int, default=32)
    parser.add_argument("--max_chain_dist", type=int, default=10000)
    
    parser.add_argument("--chain_encoder_layers", type=int, default=4)
    parser.add_argument("--chain_attention_heads", type=int, default=12)
    parser.add_argument("--chain_pair_hidden_dim", type=int, default=64)
    parser.add_argument("--chain_ffn_embed_dim", type=int, default=3072)
    parser.add_argument("--max_chain_tokens", type=int, default=1024)
    parser.add_argument("--max_segments", type=int, default=10)
    parser.add_argument("--num_block_types", type=int, default=8)
    parser.add_argument("--num_seg_smiles_types", type=int, default=20000)
    parser.add_argument(
        "--stage2_only_repr",
        choices=["full", "smiles_only", "fp_morgan2048"],
        default="full",
        help=(
            "Stage2-only segment initialization: "
            "full=dop+block+smiles+pos, "
            "smiles_only=SMILES embedding only, "
            "fp_morgan2048=fixed Morgan/ECFP4(2048) fingerprints + small projection."
        ),
    )

    # ---- Stage2-only fingerprint (Morgan/ECFP) settings ----
    parser.add_argument(
        "--fp_csv",
        default="data/raw/bcdb.csv",
        help="CSV used to rebuild dataset-level SMILES vocab and Morgan fingerprint table (Stage2-only fp_morgan2048).",
    )
    parser.add_argument(
        "--fp_smiles_prefix",
        default="SMILES",
        help="SMILES column prefix in fp_csv (expects columns like SMILES0..SMILES9).",
    )
    parser.add_argument(
        "--fp_radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (radius=2 corresponds to ECFP4).",
    )
    parser.add_argument(
        "--fp_nbits",
        type=int,
        default=2048,
        help="Morgan fingerprint bit-size.",
    )
    parser.add_argument(
        "--fp_replace_star_fallback",
        dest="fp_replace_star_fallback",
        action="store_true",
        default=True,
        help="If RDKit fails to parse SMILES, retry with '*' replaced by 'C' (enabled by default).",
    )
    parser.add_argument(
        "--no_fp_replace_star_fallback",
        dest="fp_replace_star_fallback",
        action="store_false",
        help="Disable '*'->'C' fallback when building fingerprints.",
    )
    parser.add_argument(
        "--disable_anchor_symmetry_break",
        action="store_true",
        help="Disable anchor-based symmetry breaking field in chain token construction.",
    )
    
    # ---- Target RMSE Early Stop (for finetune/reg) -------
    parser.add_argument(
        "--stop_on_target_rmse",
        action="store_true",
        help="Stop early if val RMSE reaches target within a given epoch range (finetune/reg)."
    )
    parser.add_argument("--target_rmse", type=float, default=0.4)
    parser.add_argument("--target_rmse_max_epoch", type=int, default=100)

    # ---- System Settings ----------------------------------
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--pin_memory", action="store_true")

    # ---- Distributed Training -----------------------------
    parser.add_argument("--distributed", action="store_true", help="Enable DDP")
    parser.add_argument("--dist-backend", dest="dist_backend", default="nccl")
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 1)))
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", 0)))

    # ---- Model / Task -------------------------------------
    parser.add_argument("--main_task", choices=["pretrain", "finetune", "chain", "stage2_only"], default="finetune")
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
    parser.add_argument("--droppath_prob", type=float, default=0)
    parser.add_argument("--pair_dropout", type=float, default=0.1)
    parser.add_argument("--num_pair", type=int, default=512)
    parser.add_argument("--num_kernel", type=int, default=128)
    parser.add_argument("--gaussian_std_width", type=float, default=1.0)
    parser.add_argument("--gaussian_mean_start", type=float, default=0.0)
    parser.add_argument("--gaussian_mean_stop", type=float, default=9.0)

    return parser.parse_args()

# ==================================================
#  Utilities
# ==================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(args):
    """Initialize torch.distributed"""
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        assert dist.is_initialized()
    else:
        args.rank = 0
        args.world_size = 1


def is_main_process(rank: int) -> bool:
    return rank == 0


def save_json(obj: Dict[str, Any], path: Path):
    def convert(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.int32, np.int64)):
            return int(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)


def save_checkpoint_pretrain(state: Dict[str, Any], ckpt_dir: Path, step: int):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_{step}.pt"
    torch.save(state, ckpt_path)

def save_checkpoint_finetune(state: Dict[str, Any], ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint.pt"
    torch.save(state, ckpt_path)


def load_monomer_pretrained_weights(model, weight_path, args):
    ckpt = torch.load(weight_path, map_location="cpu")

    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model_dict = model.state_dict()

    allowed_prefixes = (
        "embed_tokens.",
        "atom_feature.",
        "edge_feature.",
        "encoder.",
        "se3_invariant_kernel.",
        "lm_head.",                
        "movement_pred_head.",     
    )

    skip_prefixes = (
        "reg_head.",
        "chain_token_feature.",
        "chain_edge_feature.",
        "chain_encoder.",
    )

    load_dict = {}
    for k, v in state_dict.items():
        if not k in model_dict:
            continue
        if any(k.startswith(p) for p in skip_prefixes):
            continue
        if any(k.startswith(p) for p in allowed_prefixes):
            if model_dict[k].shape == v.shape:
                load_dict[k] = v

    missing, unexpected = model.load_state_dict(load_dict, strict=False)

    if args.rank == 0:
        print("====== Loaded monomer-level pretrained weights ======")
        print(f"Loaded keys: {len(load_dict)}")
        print(f"Missing keys (expected): {missing}")
        print(f"Unexpected keys (ignored): {unexpected}")
        print("====================================================")

    return model


def freeze_all_but_head(model):

    m = model.module if hasattr(model, "module") else model

    # 1) freeze all
    for p in m.parameters():
        p.requires_grad = False

    # 2) unfreeze head
    for p in m.reg_head.parameters():
        p.requires_grad = True

    # (optional but recommended) set frozen modules to eval to disable dropout, etc.
    # Keep head in train mode.
    m.embed_tokens.eval()
    m.atom_feature.eval()
    m.edge_feature.eval()
    m.encoder.eval()
    m.se3_invariant_kernel.eval()

    m.reg_head.train()


def freeze_for_chain_stage1(model):
    """
    Stage-1 chain training:
    - Freeze monomer-level encoder
    - Train chain-level encoder + reg_head
    """
    m = model.module if hasattr(model, "module") else model

    # ---- freeze everything first ----
    for p in m.parameters():
        p.requires_grad = False

    # ---- unfreeze chain-level modules ----
    for p in m.chain_token_feature.parameters():
        p.requires_grad = True
    for p in m.chain_edge_feature.parameters():
        p.requires_grad = True
    for p in m.chain_encoder.parameters():
        p.requires_grad = True
    for p in m.reg_head.parameters():
        p.requires_grad = True

    # ---- eval frozen parts ----
    m.embed_tokens.eval()
    m.atom_feature.eval()
    m.edge_feature.eval()
    m.encoder.eval()
    m.se3_invariant_kernel.eval()

    # ---- train chain parts ----
    m.chain_token_feature.train()
    m.chain_edge_feature.train()
    m.chain_encoder.train()
    m.reg_head.train()


def apply_pkl_metadata_to_args(args):
    """Best-effort: align model hyperparameters to dataset metadata.

    This is especially important for Stage-2-only SMILES-type embeddings:
    if num_seg_smiles_types is left at a large default, it will distort the
    Stage2big active-parameter equalization baseline.
    """
    if getattr(args, "pkl_path", None) is None:
        return

    try:
        with open(args.pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return

    smiles_vocab_size = data.get("smiles_vocab_size", None)
    if smiles_vocab_size is not None:
        smiles_vocab_size = int(smiles_vocab_size)
        # If user kept default, override; otherwise ensure it's large enough.
        if getattr(args, "num_seg_smiles_types", 20000) == 20000:
            args.num_seg_smiles_types = smiles_vocab_size
        else:
            args.num_seg_smiles_types = max(int(args.num_seg_smiles_types), smiles_vocab_size)

    max_segments = data.get("max_segments", None)
    if max_segments is not None:
        max_segments = int(max_segments)
        if getattr(args, "max_segments", 10) == 10:
            args.max_segments = max_segments
        else:
            args.max_segments = max(int(args.max_segments), max_segments)

    label_names = data.get("label_names", None)
    if label_names is not None:
        args.label_names = list(label_names)

    if getattr(args, "rank", 0) == 0:
        msg = (
            f"[META] pkl={args.pkl_path} smiles_vocab_size={smiles_vocab_size} "
            f"-> num_seg_smiles_types={getattr(args, 'num_seg_smiles_types', None)}; "
            f"max_segments(meta)={max_segments} -> max_segments={getattr(args, 'max_segments', None)}"
        )
        print(msg)


def maybe_mask_temperature_feature(batch, args):
    """Optionally remove temperature channels from batch features in-place."""
    if not getattr(args, "mask_temperature_feature", False):
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


# ==================================================
#  Regression Target Normalization (MD-style)
# ==================================================
def _parse_comma_list(v: str) -> list[str]:
    if v is None:
        return []
    v = str(v).strip()
    if not v:
        return []
    return [s.strip() for s in v.split(",") if s.strip()]


def _label_transform_torch(
    y_raw: torch.Tensor,
    label_names: list[str],
    *,
    log10_labels: set[str],
    clamp_min: float,
) -> torch.Tensor:
    if not log10_labels:
        return y_raw
    y = y_raw.clone()
    for i, name in enumerate(label_names):
        if name in log10_labels:
            y[:, i] = torch.log10(torch.clamp(y[:, i], min=float(clamp_min)))
    return y


def _label_inverse_transform_torch(
    y_trans: torch.Tensor,
    label_names: list[str],
    *,
    log10_labels: set[str],
) -> torch.Tensor:
    if not log10_labels:
        return y_trans
    y = y_trans.clone()
    for i, name in enumerate(label_names):
        if name in log10_labels:
            y[:, i] = torch.pow(10.0, y[:, i])
    return y


def _compute_label_norm_from_train_dataset(
    train_dataset,
    label_names: list[str],
    log10_labels: list[str],
    clamp_min: float,
) -> dict[str, object]:
    ys = []
    for sid in getattr(train_dataset, "ids", list(range(len(train_dataset)))):
        sample = train_dataset._get_raw_sample(sid) if hasattr(train_dataset, "_get_raw_sample") else train_dataset[sid]
        lbl = sample.get("label", None)
        if isinstance(lbl, dict):
            row = [lbl.get(k, None) for k in label_names]
        else:
            row = list(lbl)
        ys.append(row)

    y_raw = np.array(ys, dtype=np.float64)
    if np.isnan(y_raw).any():
        raise ValueError("NaN found in training labels; please clean/preprocess dataset")

    log10_set = set(log10_labels or [])
    y_trans = y_raw.copy()
    for i, name in enumerate(label_names):
        if name in log10_set:
            y_trans[:, i] = np.log10(np.clip(y_trans[:, i], a_min=float(clamp_min), a_max=None))

    mean = y_trans.mean(axis=0)
    std = y_trans.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    return {
        "label_names": list(label_names),
        "log10_labels": sorted([k for k in label_names if k in log10_set]),
        "log10_clamp_min": float(clamp_min),
        "mean": mean.astype(np.float64).tolist(),
        "std": std.astype(np.float64).tolist(),
    }


# ==================================================
#  Training and Evaluation
# ==================================================
def train_one_epoch(model, loader, optimizer, scaler, epoch, args, sampler, label_norm=None, label_names=None):
    model.train()
    
    mean_torch = None
    std_torch = None
    clamp_min = None
    log10_labels = None
    if label_norm is not None and label_names is not None and args.task_type == "reg":
        mean_torch = torch.tensor(label_norm["mean"], device="cuda", dtype=torch.float32)
        std_torch = torch.tensor(label_norm["std"], device="cuda", dtype=torch.float32)
        clamp_min = float(label_norm.get("log10_clamp_min", getattr(args, "log10_clamp_min", 1e-16)))
        log10_labels = set(label_norm.get("log10_labels", []))

    if getattr(args, "freeze_encoder", False):
        m = model.module if hasattr(model, "module") else model
        m.embed_tokens.eval()
        m.atom_feature.eval()
        m.edge_feature.eval()
        m.encoder.eval()
        m.se3_invariant_kernel.eval()
        m.reg_head.train()

    if sampler is not None:
        sampler.set_epoch(epoch)

    running_loss = 0.0
    total_samples = 0

    for step, batch in tqdm(
        enumerate(loader),
        total=len(loader),
        disable=not is_main_process(args.rank),
        desc=f"Epoch {epoch}"
    ):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)
        maybe_mask_temperature_feature(batch, args)

        with autocast(enabled=args.amp):
            # ---------- 1. Forward ----------
            pred = model(batch)
            #if torch.isnan(pred).any():
                #print(f"[Step {step}] NaN detected in model output (forward)")
                #raise ValueError("NaN in forward output")
            
            # ---------- 2. Loss ----------
            if args.task_type == "reg":
                raw_label = batch["label"]
                label = raw_label.float().cuda()

                if mean_torch is not None and std_torch is not None:
                    label_t = _label_transform_torch(
                        label,
                        label_names,
                        log10_labels=log10_labels,
                        clamp_min=clamp_min,
                    )
                    label = (label_t - mean_torch) / std_torch

                loss = F.mse_loss(pred, label) / args.grad_accum_steps
            else:
                if args.task_type == "cls":
                    raw_label = batch["label"].squeeze(-1)
                    label = raw_label.long().cuda()  
                loss = F.cross_entropy(pred, label) / args.grad_accum_steps
            #if torch.isnan(loss).any():
                #print(f"[Step {step}] NaN detected in loss computation")
                #raise ValueError("NaN in loss")

        # ---------- 3. Backward ----------
        scaler.scale(loss).backward()

        # ---------- 4. Check gradients ----------
        #for name, param in model.named_parameters():
            #if param.grad is not None and torch.isnan(param.grad).any():
                #print(f"[Step {step}] NaN in gradient of {name}")
                #raise ValueError(f"NaN in gradient: {name}")

        # ---------- 5. Optimizer step ----------
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


@torch.no_grad()
def evaluate(model, loader, args, label_norm=None, label_names=None):
    model.eval()
    
    if args.task_type == "reg":
        sum_abs_z = 0.0
        sum_sq_z = 0.0
        sum_label_z = 0.0
        sum_label_z_sq = 0.0
        sum_abs_raw = 0.0
        sum_sq_raw = 0.0
        sum_label_raw = 0.0
        sum_label_raw_sq = 0.0
        count = 0
        all_residuals = []  # store z residuals if normalized, else raw residuals
    else:
        conf_matrix = None
        C = args.num_tasks
        conf_matrix = np.zeros((C, C), dtype=np.int64)
        all_preds = []
        all_labels = []

    mean_t = None
    std_t = None
    clamp_min = None
    log10_labels = None
    use_norm = bool(
        args.task_type == "reg"
        and label_norm is not None
        and label_names is not None
        and getattr(args, "label_zscore", False)
    )
    if use_norm:
        mean_t = torch.tensor(label_norm["mean"], device="cuda", dtype=torch.float32)
        std_t = torch.tensor(label_norm["std"], device="cuda", dtype=torch.float32)
        clamp_min = float(label_norm.get("log10_clamp_min", getattr(args, "log10_clamp_min", 1e-16)))
        log10_labels = set(label_norm.get("log10_labels", []))

    for batch in loader:
        # Move batch to CUDA
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)
        maybe_mask_temperature_feature(batch, args)

        # Forward pass
        pred = model(batch)
        label_raw = batch["label"].float().cuda()

        if args.task_type == "reg":
            if use_norm:
                label_trans = _label_transform_torch(
                    label_raw,
                    label_names,
                    log10_labels=log10_labels,
                    clamp_min=clamp_min,
                )
                label_z = (label_trans - mean_t) / std_t
                pred_z = pred
                residual_z = (pred_z - label_z).detach().cpu().numpy().flatten()

                pred_trans = pred_z * std_t + mean_t
                pred_raw_proc = _label_inverse_transform_torch(pred_trans, label_names, log10_labels=log10_labels)
                label_raw_proc = _label_inverse_transform_torch(label_trans, label_names, log10_labels=log10_labels)
                residual_raw = (pred_raw_proc - label_raw_proc).detach().cpu().numpy().flatten()

                if is_main_process(args.rank):
                    all_residuals.append(residual_z)

                sum_abs_z += np.abs(residual_z).sum()
                sum_sq_z += (residual_z ** 2).sum()
                label_z_cpu = label_z.detach().cpu().numpy().flatten()
                sum_label_z += label_z_cpu.sum()
                sum_label_z_sq += (label_z_cpu ** 2).sum()

                sum_abs_raw += np.abs(residual_raw).sum()
                sum_sq_raw += (residual_raw ** 2).sum()
                label_raw_cpu = label_raw_proc.detach().cpu().numpy().flatten()
                sum_label_raw += label_raw_cpu.sum()
                sum_label_raw_sq += (label_raw_cpu ** 2).sum()
                count += len(residual_z)

            else:
                pred_cpu = pred.detach().cpu().numpy().flatten()
                label_cpu = label_raw.detach().cpu().numpy().flatten()
                residuals = pred_cpu - label_cpu

                if is_main_process(args.rank):
                    all_residuals.append(residuals)

                sum_abs_raw += np.abs(residuals).sum()
                sum_sq_raw += (residuals ** 2).sum()
                sum_label_raw += label_cpu.sum()
                sum_label_raw_sq += (label_cpu ** 2).sum()
                count += len(residuals)
        else:
            pred_cpu = pred.cpu().numpy()
            label_cpu = label_raw.cpu().numpy().flatten()
            pred_cls = pred_cpu.argmax(axis=-1).flatten()
            
            all_preds.append(pred_cls)
            all_labels.append(label_cpu)

            for t, p in zip(label_cpu, pred_cls):
                conf_matrix[int(t), int(p)] += 1

    if args.task_type == "reg":
        if args.distributed:
            stats = torch.tensor(
                [
                    sum_abs_z, sum_sq_z, sum_label_z, sum_label_z_sq,
                    sum_abs_raw, sum_sq_raw, sum_label_raw, sum_label_raw_sq,
                    count,
                ],
                device="cuda",
                dtype=torch.float32,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            (
                sum_abs_z, sum_sq_z, sum_label_z, sum_label_z_sq,
                sum_abs_raw, sum_sq_raw, sum_label_raw, sum_label_raw_sq,
                count,
            ) = stats.cpu().numpy()
        
        if count == 0:
            return {"MAE": 0, "RMSE": 0, "R2": 0, "residuals": None}
        
        mae_raw = sum_abs_raw / count
        rmse_raw = np.sqrt(sum_sq_raw / count)
        total_var_raw = sum_label_raw_sq - (sum_label_raw ** 2) / count
        r2_raw = 1.0 - (sum_sq_raw / total_var_raw) if total_var_raw > 1e-7 else 0.0

        if label_norm is not None and label_names is not None and getattr(args, "label_zscore", False):
            mae_z = sum_abs_z / count
            rmse_z = np.sqrt(sum_sq_z / count)
            total_var_z = sum_label_z_sq - (sum_label_z ** 2) / count
            r2_z = 1.0 - (sum_sq_z / total_var_z) if total_var_z > 1e-7 else 0.0
        else:
            mae_z = mae_raw
            rmse_z = rmse_raw
            r2_z = r2_raw
        
        residuals_concat = np.concatenate(all_residuals) if is_main_process(args.rank) else None
        return {
            "MAE": mae_z,
            "RMSE": rmse_z,
            "R2": r2_z,
            "MAE_raw": mae_raw,
            "RMSE_raw": rmse_raw,
            "R2_raw": r2_raw,
            "residuals": residuals_concat
        }
    else:
        if args.distributed and conf_matrix is not None:
            tensor_conf = torch.tensor(conf_matrix, device="cuda", dtype=torch.int64)
            dist.all_reduce(tensor_conf, op=dist.ReduceOp.SUM)
            conf_matrix = tensor_conf.cpu().numpy()

        if conf_matrix is None:
            conf_matrix = np.zeros((1, 1))
            acc = 0.0
        else:
            correct = conf_matrix.trace()
            total = conf_matrix.sum()
            acc = correct / total if total > 0 else 0.0
        
        preds_concat = np.concatenate(all_preds) if is_main_process(args.rank) else None
        labels_concat = np.concatenate(all_labels) if is_main_process(args.rank) else None
        return {
            "ACC": acc,
            "confusion": conf_matrix,
            "preds": preds_concat,
            "labels": labels_concat
        }


@torch.no_grad()
def evaluate_pretrain(model, loader, args):
    model.eval()
    running = dict(
        Latom_sum=0.0, Lcoord_sum=0.0, Ldist_sum=0.0,
        n_masked=0, n_atoms=0, n_pairs=0
    )

    for batch in loader:
        batch = {
            k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        maybe_mask_temperature_feature(batch, args)
        logits, pred_pos, pred_dist = model(batch)

        # ---- Latom ----
        tgt_tok = batch["target_token"]  # (B, N)
        mask_tok = tgt_tok.ne(0)         # (B, N)
        if mask_tok.any():
            L_atom = F.cross_entropy(logits[mask_tok], tgt_tok[mask_tok], reduction="mean")
        else:
            L_atom = logits.new_tensor(0.0)

        # ---- Lcoord ----
        atom_mask = batch["atom_mask"].bool()    # (B, N)
        pos_mask = atom_mask.unsqueeze(-1)       # (B, N, 1)
        tgt_pos = batch["target_pos"].float()    # (B, N, 3)
        diff_pos = (pred_pos - tgt_pos).abs() * pos_mask
        coord_per_mol = diff_pos.sum((-1, -2)) / (pos_mask.sum((-1, -2)) + 1e-10)
        L_coord = coord_per_mol.mean()

        # ---- Ldist ----
        pair_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)  # (B, N, N)
        tgt_dist = torch.cdist(tgt_pos, tgt_pos)                       # (B, N, N)
        diff_dist = (pred_dist - tgt_dist).abs() * pair_mask
        dist_per_mol = diff_dist.sum((-1, -2)) / (pair_mask.sum((-1, -2)) + 1e-10)
        L_dist = dist_per_mol.mean()

        # ---- Accumulate ----
        n_masked = mask_tok.sum().item()
        n_atoms = atom_mask.sum().item()
        n_pairs = pair_mask.sum().item()

        running["Latom_sum"] += L_atom.item() * n_masked
        running["Lcoord_sum"] += L_coord.item() * n_atoms
        running["Ldist_sum"] += L_dist.item() * max(n_pairs, 1)
        running["n_masked"] += n_masked
        running["n_atoms"] += n_atoms
        running["n_pairs"] += n_pairs

    # ---- DDP sync (optional) ----
    if args.distributed:
        keys = ["Latom_sum", "Lcoord_sum", "Ldist_sum", "n_masked", "n_atoms", "n_pairs"]
        tensors = [torch.tensor(running[k], device="cuda") for k in keys]
        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for k, t in zip(keys, tensors):
            running[k] = t.item()

    # ---- Compute average ----
    avg_Latom = running["Latom_sum"] / max(running["n_masked"], 1)
    avg_Lcoord = running["Lcoord_sum"] / max(running["n_atoms"], 1)
    avg_Ldist = running["Ldist_sum"] / max(running["n_pairs"], 1)

    return {
        "Latom": avg_Latom,
        "Lcoord": avg_Lcoord,
        "Ldist": avg_Ldist
    }


# ==================================================
#  Main Function
# ==================================================
def main():
    args = parse_args()
    args.distributed = args.world_size > 1 or args.distributed
    set_seed(args.seed)
    init_distributed(args)
    best_epoch = 0    

    # Align Stage-2-only SMILES embedding sizes to the dataset vocabulary.
    apply_pkl_metadata_to_args(args)

    model = MultiMolModel(args)
        
    # ============ Block for finetune task & Directory Setup ============
    if args.main_task in {"finetune", "chain", "stage2_only"}:
        is_unimol2 = (args.weight_path is not None and "unimol2" in os.path.basename(args.weight_path))
        if args.weight_path is not None:
            try:
                model = load_monomer_pretrained_weights(model, args.weight_path, args)
            except FileNotFoundError:
                print("Checkpoint file not found, training from scratch.")
        if args.freeze_encoder:
            freeze_for_chain_stage1(model)    
        model = model.cuda()
        if args.distributed:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,)
        pkl_tag = Path(args.pkl_path).stem if args.pkl_path else "nopkl"
        wt_tag = Path(args.weight_path).stem if args.weight_path else "nowt"
        run_stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = (
            Path(args.results_root)
            / args.dataset_name
            / f"task_{args.main_task}_eq_{int(args.equalize_active_params)}_freeze_{int(args.freeze_encoder)}"
            / f"data_{pkl_tag}"
            / f"wt_{wt_tag}"
        )

        if args.main_task in {"chain", "stage2_only"}:
            run_dir = run_dir / (
                f"repr_{args.stage2_only_repr}_noanchor_{int(args.disable_anchor_symmetry_break)}"
            )

        run_dir = run_dir / f"fold_{args.fold}" / (
            f"wd_{args.weight_decay}_lr_{args.lr}_do_{args.dropout}_wogroup_{args.wo_pair}"
        )
        run_dir = run_dir / f"run_{run_stamp}"
        ckpt_dir = run_dir / "checkpoints"
        log_dir = run_dir / "logs"
        pred_dir = run_dir / "predictions"
        for d in (ckpt_dir, log_dir, pred_dir):
            if is_main_process(args.rank):
                d.mkdir(parents=True, exist_ok=True)
    
        # ============ Data Loaders ============            
        if not args.non_kfold:
            train_loader, train_sampler = build_dataloader(args.pkl_path, args, mode="train")
            val_loader, _ = build_dataloader(args.pkl_path, args, mode="val")
        else:
            train_loader, train_sampler = build_dataloader(args.pkl_path, args, mode="train")
            val_loader, _ = build_dataloader(args.test_pkl_path, args, mode="full")

        # ============ Label normalization (regression only) ============
        label_norm = None
        label_names = list(getattr(args, "label_names", [])) if getattr(args, "label_zscore", False) else None
        if args.task_type == "reg" and getattr(args, "label_zscore", False):
            if not label_names:
                try:
                    with open(args.pkl_path, "rb") as f:
                        header = pickle.load(f)
                    label_names = list(header.get("label_names", []))
                except Exception:
                    label_names = []

            if not label_names:
                raise ValueError("--label_zscore requires label_names in pkl header (missing label_names).")

            requested_log10 = _parse_comma_list(getattr(args, "log10_labels", ""))
            log10_labels = [k for k in requested_log10 if k in set(label_names)]
            unknown = [k for k in requested_log10 if k not in set(label_names)]
            if unknown and is_main_process(args.rank):
                print(f"[WARN] log10_labels not in label_names and will be ignored: {unknown}")

            label_norm = _compute_label_norm_from_train_dataset(
                train_loader.dataset,
                label_names=label_names,
                log10_labels=log10_labels,
                clamp_min=float(getattr(args, "log10_clamp_min", 1e-16)),
            )
            if is_main_process(args.rank):
                save_json(label_norm, run_dir / "label_norm.json")

    
        # ============ Optimizer & Scheduler ============
        #optimizer = torch.optim.AdamW(
        #    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        #)
        m = model.module if isinstance(model, DDP) else model


        if args.main_task == "chain" and args.freeze_encoder:
            optimizer = torch.optim.AdamW(
                [
                    *m.chain_token_feature.parameters(),
                    *m.chain_edge_feature.parameters(),
                    *m.chain_encoder.parameters(),
                    *m.reg_head.parameters(),
                ],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            
        scaler = GradScaler(enabled=args.amp)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #    optimizer,
        #    T_0=20,
        #    T_mult=1,
        #    eta_min=args.lr * 0.01
        #)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=40, eta_min=args.lr * 0.01
        )
        # ============ Training Loop ============
        metrics_history = {}
        best_score = float("inf") if args.task_type == "reg" else 0.0
        
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                epoch,
                args,
                train_sampler,
                label_norm=label_norm,
                label_names=label_names,
            )
            scheduler.step()
        
            val_metrics = evaluate(model, val_loader, args, label_norm=label_norm, label_names=label_names)
            epoch_time = time.time() - epoch_start
            peak_mem_gb = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
            if is_main_process(args.rank):
                # ----- Logging & Printing -----
                if args.task_type == "reg":
                    peak_mem_suffix = (
                        f" peak_mem_gb={peak_mem_gb:.3f}"
                        if peak_mem_gb is not None
                        else ""
                    )
                    print(
                        f"[E{epoch:03d}] "
                        f"train_loss={train_loss:.{METRIC_PRINT_DIGITS}f} "
                        f"val_mae={val_metrics['MAE']:.{METRIC_PRINT_DIGITS}f} "
                        f"val_rmse={val_metrics['RMSE']:.{METRIC_PRINT_DIGITS}f} "
                        f"val_r2={val_metrics['R2']:.{METRIC_PRINT_DIGITS}f} "
                        f"val_rmse_raw={val_metrics.get('RMSE_raw', val_metrics['RMSE']):.{METRIC_PRINT_DIGITS}f} "
                        f"val_r2_raw={val_metrics.get('R2_raw', val_metrics['R2']):.{METRIC_PRINT_DIGITS}f} "
                        f"time={epoch_time:.1f}s"
                        f"{peak_mem_suffix}"
                    )
                else:
                    peak_mem_suffix = (
                        f" peak_mem_gb={peak_mem_gb:.3f}"
                        if peak_mem_gb is not None
                        else ""
                    )
                    print(
                        f"[E{epoch:03d}] "
                        f"train_loss={train_loss:.{METRIC_PRINT_DIGITS}f} "
                        f"val_acc={val_metrics['ACC']:.{METRIC_PRINT_DIGITS}f} "
                        f"time={epoch_time:.1f}s"
                        f"{peak_mem_suffix}"
                    )
        
                # ----- Save Metrics -----
                metrics_history[epoch] = {
                    "train_loss": train_loss,
                    **val_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                    "peak_mem_gb": peak_mem_gb,
                }
                save_json(metrics_history, run_dir / "metrics.json")
        
                # ----- Save Residuals or Confusion Matrix -----
                if args.task_type == "reg":
                    np.save(pred_dir / f"residuals_ep{epoch}.npy", val_metrics["residuals"])
                else:
                    np.save(pred_dir / f"confusion_ep{epoch}.npy", val_metrics["confusion"])
                    np.save(pred_dir / f"preds_ep{epoch}.npy", val_metrics["preds"])
                    np.save(pred_dir / f"labels_ep{epoch}.npy", val_metrics["labels"])
        
                # ----- Save Best Checkpoint -----
                is_better =  (
                    (args.task_type == "reg" and val_metrics["RMSE"] < best_score)
                    or (args.task_type == "cls" and val_metrics["ACC"] > best_score)
                )
                
                if is_better:
                    best_epoch = epoch
                    best_score = (
                        val_metrics["RMSE"] if args.task_type == "reg" else val_metrics["ACC"]
                    )
                    if args.main_task != "finetune" or args.save_finetune_checkpoint:
                        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                        save_checkpoint_finetune(
                            {
                                "epoch": epoch,
                                "model_state": model_state,
                                "optimizer_state": optimizer.state_dict(),
                                "scaler_state": scaler.state_dict(),
                                "args": vars(args),
                                "label_norm": label_norm,
                            },
                            ckpt_dir,
                        )
            if args.distributed:
                be_t = torch.tensor(best_epoch, device=args.local_rank)
                dist.broadcast(be_t, src=0)
                best_epoch = be_t.item()
            
            # ----- Target-RMSE early stop (your rule) -----
            target_stop = False
            if args.task_type == "reg" and args.stop_on_target_rmse:
                cur_rmse = float(val_metrics["RMSE"])
                if epoch <= args.target_rmse_max_epoch and cur_rmse < args.target_rmse:
                    target_stop = True

            # Make sure all ranks stop together
            if args.distributed and dist.is_initialized():
                if is_main_process(args.rank):
                    flag_tensor = torch.tensor(int(target_stop), device="cuda")
                else:
                    flag_tensor = torch.zeros(1, device="cuda")
                dist.broadcast(flag_tensor, src=0)
                target_stop = bool(flag_tensor.item())

            if target_stop:
                if is_main_process(args.rank):
                    print(
                        f"Target-RMSE early stop triggered at epoch {epoch}: "
                        f"val_rmse={val_metrics['RMSE']:.4f} < {args.target_rmse} "
                        f"(max_epoch={args.target_rmse_max_epoch})"
                    )
                break
            
            if args.early_stop_patience and (epoch - best_epoch >= args.early_stop_patience):
                if is_main_process(args.rank):
                    print(f"Early stopping triggered at epoch {epoch}, best was epoch {best_epoch}, best score was {best_score}")
                break
        
        if args.distributed:
            dist.destroy_process_group()
    
    # ============ Block for pretrain task ============
    if args.main_task == "pretrain":
    
        # -------- Dataloader & model --------
        model = model.cuda()
        train_path = args.pretrain_train_path
        val_path   = args.pretrain_val_path
        if args.pretrain_use_split:
            pre_loader, pre_sampler = build_dataloader(train_path, args, mode="train")
            val_loader, _ = build_dataloader(
                train_path if os.path.abspath(train_path) == os.path.abspath(val_path) else val_path,
                args,
                mode="val",
            )
        else:
            pre_loader, pre_sampler = build_dataloader(train_path, args, mode="full")
            val_loader, _ = build_dataloader(val_path, args, mode="full")
    
        # -------- Directory setup --------
        pretrain_data_tag = Path(train_path).stem if train_path else "nopkl"
        split_tag = (
            f"split_fold{int(args.pretrain_split_fold)}"
            if args.pretrain_use_split
            else "split_full"
        )
        run_dir  = (
            Path(args.results_root)
            / "random_pretrain"
            / f"data_{pretrain_data_tag}"
            / split_tag
            / f"lr_{args.lr}_wd_{args.weight_decay}_bs_{args.batch_size}"
        )
        ckpt_dir = run_dir / "checkpoints"
        log_dir  = run_dir / "logs"
        if is_main_process(args.rank):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True,  exist_ok=True)
    
        # -------- Optimizer & Scaler --------
        eff_bs   = args.batch_size * args.world_size * args.grad_accum_steps
        base_lr  = args.lr
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=args.weight_decay
        )
        dataset_size   = len(pre_loader.dataset)          
        steps_per_epoch = math.ceil(dataset_size / eff_bs)
        total_steps     = steps_per_epoch * args.epochs
        warmup = (
            args.warmup_steps if args.warmup_steps > 0
            else max(1000, int(total_steps * args.warmup_ratio))
        )
    
        def lr_lambda(step: int):
            # Linear warm-up + cosine decay
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = GradScaler(enabled=args.amp)
    
        # -------- Resume from checkpoint or initialize --------
        ckpt_path = run_dir / "checkpoints" / "checkpoint_randomrandom.pt"
        no_random = False
        if no_random:
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=f"cuda:{args.local_rank}")["model"]        
            model_dict = model.state_dict()
            skip_prefixes = (
                "se3_invariant_kernel.gaussian.mul",
                "se3_invariant_kernel.gaussian.bias",
                "reg_head",
            )        
            for name, param in state_dict.items():
                if name not in model_dict:
                    continue
                tgt = model_dict[name]
                if param.shape != tgt.shape:
                    if name.startswith(skip_prefixes):
                        print(f" Skipping layer {name!r}: checkpoint {param.shape} != model {tgt.shape}")
                        continue
                    raise RuntimeError(
                        f"Shape mismatch for {name!r}: checkpoint {param.shape} vs model {tgt.shape}"
                    )
                model_dict[name] = param        
            model.load_state_dict(model_dict, strict=False)        
            start_epoch = 1
            global_step = 0
            best_loss = None
            best_step = 0
            intervals_no_improve = 0            
            
        else:
            if ckpt_path.is_file():
                ckpt = torch.load(ckpt_path, map_location=f"cuda:{args.local_rank}")
                missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
                if missing or unexpected:
                    print(f"[Resume][WARN] strict=False missing={len(missing)} unexpected={len(unexpected)}")
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scaler.load_state_dict(ckpt["scaler_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                start_epoch      = ckpt["epoch"]
                global_step      = ckpt["global_step"]
                best_loss        = ckpt["best_loss"]
                best_step        = ckpt["best_step"]
                intervals_no_improve = ckpt["intervals_no_improve"]
                print(f"[Resume] epoch={start_epoch}  step={global_step}  best_loss={best_loss:.5f}")
            else:
                start_epoch      = 1
                global_step      = 0                     
                best_loss        = None
                best_step        = 0
                intervals_no_improve = 0
    
        # -------- DDP wrapper --------
        if args.distributed:
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )
    
        # -------- High-frequency validation --------
        val_every_steps = getattr(args, "val_every_steps", 20)
        min_delta_abs   = getattr(args, "min_delta_abs", 1e-3)
        min_delta_rel   = getattr(args, "min_delta_rel", 1e-2)
    
        # -------- Training loop --------
        stop_training = False
        for epoch in range(start_epoch, args.epochs + 1):
            if pre_sampler is not None:
                pre_sampler.set_epoch(epoch)
    
            model.train()
            pbar = tqdm(
                pre_loader,
                disable=not is_main_process(args.rank),
                desc=f"Pre-E{epoch}",
            )
    
            for step_idx, batch in enumerate(pbar):
                # -------------- Forward & Loss --------------
                batch = {
                    k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                maybe_mask_temperature_feature(batch, args)
    
                with autocast(enabled=args.amp):
                    logits, pred_pos, pred_dist = model(batch)
    
                    # ---- L_atom ----
                    tgt_tok = batch["target_token"]
                    mask_tok = tgt_tok.ne(0)
                    if mask_tok.any():
                        L_atom = F.cross_entropy(
                            logits[mask_tok], tgt_tok[mask_tok], reduction="mean"
                        )
                    else:
                        L_atom = logits.new_tensor(0.0)
    
                    # ---- L_coord ----
                    atom_mask = batch["atom_mask"].bool()
                    pos_mask  = atom_mask.unsqueeze(-1)
                    tgt_pos   = batch["target_pos"].float()
                    diff_pos  = (pred_pos - tgt_pos).abs() * pos_mask
                    coord_per_mol = diff_pos.sum((-1, -2)) / (pos_mask.sum((-1, -2)) + 1e-10)
                    L_coord = coord_per_mol.mean()
    
                    # ---- L_dist ----
                    pair_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
                    tgt_dist  = torch.cdist(tgt_pos, tgt_pos)
                    diff_dist = (pred_dist - tgt_dist).abs() * pair_mask
                    dist_per_mol = diff_dist.sum((-1, -2)) / (pair_mask.sum((-1, -2)) + 1e-10)
                    L_dist = dist_per_mol.mean()
    
                    loss = (L_atom + L_coord + L_dist) / args.grad_accum_steps
    
                # -------------- Backward --------------
                scaler.scale(loss).backward()
    
                if (step_idx + 1) % args.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
    
                    global_step += 1
    
                    # -------------- Validation & Checkpoint --------------
                    if global_step % val_every_steps == 0:
                        model.eval()
                        with torch.no_grad():
                            val_metrics = evaluate_pretrain(model, val_loader, args)
                        
                        torch.cuda.empty_cache()
                    
                        val_total = (val_metrics["Latom"]
                                     + val_metrics["Lcoord"]
                                     + val_metrics["Ldist"])
                    
                        # ---------- Rank-0: decide whether this is an improvement ----------
                        if is_main_process(args.rank):
                            print(
                                f"[Step {global_step}] quick-val "
                                f"Latom={val_metrics['Latom']:.4f} "
                                f"Lcoord={val_metrics['Lcoord']:.4f} "
                                f"Ldist={val_metrics['Ldist']:.4f} "
                                f"total={val_total:.4f}"
                            )
                    
                            if best_loss is None:
                                improve = True                       # first ever evaluation
                            else:
                                improve = (best_loss - val_total) > max(
                                    min_delta_abs,
                                    best_loss * min_delta_rel
                                )
                    
                            if improve:
                                best_loss = val_total
                                best_step = global_step
                                intervals_no_improve = 0
                    
                                model_state = (model.module.state_dict()
                                               if isinstance(model, DDP)
                                               else model.state_dict())
                                save_checkpoint_pretrain(
                                    {
                                        "epoch": epoch,
                                        "global_step": global_step,
                                        "model_state": model_state,
                                        "optimizer_state": optimizer.state_dict(),
                                        "scaler_state": scaler.state_dict(),
                                        "scheduler_state": scheduler.state_dict(),
                                        "best_loss": best_loss,
                                        "best_step": best_step,
                                        "intervals_no_improve": intervals_no_improve,
                                        "args": vars(args),
                                    },
                                    ckpt_dir,
                                    step=global_step,
                                )
                            else:
                                intervals_no_improve += 1
                    
                            # rank-0 prepares values to broadcast
                            loss_sync  = float(best_loss)
                            step_sync  = float(best_step)
                            noimp_sync = float(intervals_no_improve)
                        else:
                            # placeholder values on non-main ranks
                            loss_sync  = float("inf")
                            step_sync  = 0.0
                            noimp_sync = 0.0
                    
                        # ---------- broadcast best_* & intervals_no_improve ----------
                        if args.distributed and dist.is_initialized():
                            sync_tensor = torch.tensor(
                                [loss_sync, step_sync, noimp_sync],
                                dtype=torch.float32,
                                device="cuda",
                            )
                            dist.broadcast(sync_tensor, src=0)
                            best_loss, best_step, intervals_no_improve = sync_tensor.tolist()
                            intervals_no_improve = int(intervals_no_improve)  # cast back to int
                    
                        model.train()
                        
                        if is_main_process(args.rank):
                            EARLY_STOP_PATIENCE = args.early_stop_patience
                            early_stop_flag = intervals_no_improve >= EARLY_STOP_PATIENCE
                            flag_tensor = torch.tensor(int(early_stop_flag), device="cuda")
                        else:
                            flag_tensor = torch.zeros(1, device="cuda")

                        if args.distributed and dist.is_initialized():
                            dist.broadcast(flag_tensor, src=0)
                        stop_training = bool(flag_tensor.item())
                        
                        if stop_training:
                            if is_main_process(args.rank):
                                print(f"[Early-Stop] No improvement for {EARLY_STOP_PATIENCE} evals, "
                                      f"stop at global_step={global_step}.")
                            break
                # Exit inner loop early
                if stop_training:
                    break
    
            # Exit outer loop early
            if stop_training:
                break

    # -------- Clean up --------
    if args.distributed and dist.is_initialized():
        dist.destroy_process_group()


        
if __name__ == "__main__":
    main()
