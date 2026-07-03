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

MD_REG_LABEL_NAMES = [
    "density",
    "Rg",
    "D",
    "MSD_A2",
    "S_q_peak",
    "q_peak_invA",
    "eta0_Pa_s",
    "tau_maxwell_s",
    "omega_cross_rad_s",
]
MD_LOG10_LABELS_DEFAULT = {
    "D",
    "MSD_A2",
    "eta0_Pa_s",
    "tau_maxwell_s",
    "omega_cross_rad_s",
    "Cp",
    "K_T",
}
MD_D_CLAMP_MIN = 1e-16



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
    parser.add_argument(
        "--finetune_optimizer_stage1_only",
        action="store_true",
        help="(finetune only) Optimize only stage-1 modules + reg_head, excluding chain-level modules.",
    )
    parser.add_argument(
        "--stage1_capacity_mlp_hidden",
        type=int,
        default=0,
        help="(finetune only) Add a stage-1-only capacity MLP with hidden dim H (uses CLS rep).",
    )

    # ---- Dataset / Path -----------------------------------
    parser.add_argument("--dataset_name",help="Name of the dataset")
    parser.add_argument("--non_kfold", default=False, action="store_true", help="Use separate train/test pkl instead of CV folds")
    parser.add_argument("--pkl_path", help="Path to the *.pkl file")
    parser.add_argument("--test_pkl_path", help="Path to the *_test.pkl file")
    parser.add_argument(
        "--locked_test_pkl_path",
        default=None,
        help="Optional locked outer-test pkl for reporting test metrics based on best val checkpoint.",
    )
    parser.add_argument("--weight_path", default="/share/home/202320162823/unimacro/no_pretrain.pt")
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Skip loading monomer-level pretrained weights even if --weight_path is set.",
    )
    parser.add_argument(
        "--pretrained_stage1_only",
        action="store_true",
        help="When loading --weight_path, restrict loading to stage-1 encoder modules only.",
    )
    parser.add_argument("--pretrain_train_path", default="/public/home/202320162823/version2_pretrain/pretrain_data/pretrain_train.lmdb")
    parser.add_argument("--pretrain_val_path",default="/public/home/202320162823/version2_pretrain/pretrain_data/pretrain_val.lmdb")
    parser.add_argument("--fold", type=int, help="Cross-validation fold index")
    parser.add_argument("--results_root", default="results", help="Root directory to save results")

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
    parser.add_argument(
        "--early_stop_start_epoch",
        type=int,
        default=1,
        help="Start counting finetune/chain early-stop patience only from this epoch.",
    )
    parser.add_argument("--val_every_steps", type=int, default=50)
    parser.add_argument("--min_delta_abs", type=float, default=1e-3)
    parser.add_argument("--min_delta_rel", type=float, default=1e-2)
    
    # ---- Chain-level model args ----
    parser.add_argument("--num_chain_glob_feat", type=int, default=3)   # T, Mn, coexistence
    parser.add_argument("--num_chain_block_feat", type=int, default=1)  # volume fraction
    parser.add_argument("--num_seg_smiles_types", type=int, default=20000)
    parser.add_argument(
        "--chain_only_repr",
        choices=["smiles_embed", "fp_morgan2048"],
        default="smiles_embed",
        help="(chain_only) segment representation source",
    )
    parser.add_argument("--fp_radius", type=int, default=2, help="(chain_only fp) Morgan radius (radius=2 -> ECFP4)")
    parser.add_argument("--fp_nbits", type=int, default=2048, help="(chain_only fp) fingerprint bits")
    parser.add_argument(
        "--fp_replace_star_fallback",
        action="store_true",
        default=True,
        help="(chain_only fp) if RDKit fails, try smiles.replace('*','C') once",
    )
    
    parser.add_argument("--chain_pair_dim", type=int, default=32)
    parser.add_argument("--max_chain_dist", type=int, default=10000)
    
    parser.add_argument("--chain_encoder_layers", type=int, default=4)
    parser.add_argument("--chain_attention_heads", type=int, default=12)
    parser.add_argument("--chain_pair_hidden_dim", type=int, default=64)
    parser.add_argument("--chain_ffn_embed_dim", type=int, default=3072)
    parser.add_argument(
        "--disable_chain_symmetry_break",
        action="store_true",
        help="Disable stage-2 topology-anchor symmetry-breaking positional rewrite on chain node tokens.",
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
    parser.add_argument("--main_task", choices=["pretrain", "finetune", "chain", "chain_only"], default="finetune")
    parser.add_argument("--task_type", choices=["reg", "cls"], default="reg")
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument(
        "--label_index",
        type=int,
        default=-1,
        help="For regression with multi-label pkl, select one label index (0-based). -1 keeps all labels.",
    )
    parser.add_argument(
        "--log10_labels",
        type=str,
        default=",".join(sorted(MD_LOG10_LABELS_DEFAULT)),
        help="(reg) Comma-separated label names to apply log10 transform (before per-fold z-score).",
    )
    parser.add_argument(
        "--d_clamp_min",
        type=float,
        default=MD_D_CLAMP_MIN,
        help="(reg) Clamp minimum applied before log10.",
    )
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


def _parse_log10_labels_arg(s: str):
    if s is None:
        return set()
    items = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            items.append(x)
    return set(items)


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
    )

    if not getattr(args, "pretrained_stage1_only", False):
        allowed_prefixes = allowed_prefixes + (
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
    if hasattr(m, "seg_only_projs"):
        for p in m.seg_only_projs.parameters():
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
    if hasattr(m, "seg_only_projs"):
        m.seg_only_projs.train()
    m.reg_head.train()


def freeze_pretrain_heads(model):
    """Freeze pretrain-only heads (unused in finetune/chain tasks)."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "lm_head"):
        # NOTE: MaskLMHead stores a *tied* reference to embed_tokens.weight in `lm_head.weight`.
        # Freezing `lm_head.parameters()` would inadvertently freeze the embedding table.
        for p in m.lm_head.dense.parameters():
            p.requires_grad = False
        for p in m.lm_head.layer_norm.parameters():
            p.requires_grad = False
        if hasattr(m.lm_head, "bias") and isinstance(m.lm_head.bias, torch.nn.Parameter):
            m.lm_head.bias.requires_grad = False
        m.lm_head.eval()
    if hasattr(m, "movement_pred_head"):
        for p in m.movement_pred_head.parameters():
            p.requires_grad = False
        m.movement_pred_head.eval()


# ==================================================
#  Label transform / normalization (multi-task reg)
# ==================================================
def _resolve_label_names_from_dataset(dataset, args):
    raw_label_names = getattr(dataset, "label_names", None)
    if raw_label_names:
        raw_label_names = list(raw_label_names)
    else:
        raw_label_names = list(MD_REG_LABEL_NAMES[: int(getattr(args, "num_tasks", 1))])

    li = int(getattr(args, "label_index", -1))
    if li >= 0:
        # Single-label mode: pick one label by index.
        if int(getattr(args, "num_tasks", 1)) != 1:
            raise ValueError("--label_index requires --num_tasks=1 for single-label training")
        if raw_label_names and 0 <= li < len(raw_label_names):
            return [str(raw_label_names[li])]
        if 0 <= li < len(MD_REG_LABEL_NAMES):
            return [str(MD_REG_LABEL_NAMES[li])]
        raise ValueError(f"label_index={li} out of range")

    label_names = raw_label_names

    if int(getattr(args, "num_tasks", 1)) == len(MD_REG_LABEL_NAMES) and label_names != MD_REG_LABEL_NAMES:
        raise ValueError(
            f"Expected label_names={MD_REG_LABEL_NAMES}, got {label_names}. "
            "Please preprocess with explicit --labels in the expected order."
        )
    if len(label_names) != int(getattr(args, "num_tasks", 1)):
        raise ValueError(f"label_names length mismatch: len(label_names)={len(label_names)} num_tasks={args.num_tasks}")

    return label_names


def _label_transform_torch(
    y_raw: torch.Tensor,
    label_names,
    log10_labels,
    d_clamp_min: float = MD_D_CLAMP_MIN,
) -> torch.Tensor:
    y = y_raw
    if y.dim() == 1:
        y = y.unsqueeze(-1)

    y = y.clone()
    log10_labels = set(log10_labels or [])
    for i, name in enumerate(label_names):
        if name in log10_labels:
            y[:, i] = torch.log10(torch.clamp(y[:, i], min=float(d_clamp_min)))
    return y


def _label_inverse_transform_np(y_trans: np.ndarray, label_names, log10_labels) -> np.ndarray:
    y = np.array(y_trans, dtype=np.float64, copy=True)
    log10_labels = set(log10_labels or [])
    for i, name in enumerate(label_names):
        if name in log10_labels:
            y[:, i] = np.power(10.0, y[:, i])
    return y


def _compute_label_norm_from_train_dataset(
    train_dataset,
    label_names,
    log10_labels,
    d_clamp_min: float = MD_D_CLAMP_MIN,
):
    ys = []
    for sid in getattr(train_dataset, "ids", list(range(len(train_dataset)))):
        sample = train_dataset._get_raw_sample(sid) if hasattr(train_dataset, "_get_raw_sample") else train_dataset[sid]
        lbl = sample.get("label", None)
        if isinstance(lbl, dict):
            row = [lbl.get(k, None) for k in label_names]
        else:
            # Already a tensor/array-like label vector
            row = list(lbl)
        ys.append(row)

    y_raw = np.array(ys, dtype=np.float64)
    if np.isnan(y_raw).any():
        raise ValueError("NaN found in training labels; please clean/preprocess dataset")

    y_trans = y_raw.copy()
    log10_labels = set(log10_labels or [])
    for i, name in enumerate(label_names):
        if name in log10_labels:
            y_trans[:, i] = np.log10(np.clip(y_trans[:, i], a_min=float(d_clamp_min), a_max=None))

    mean = y_trans.mean(axis=0)
    std = y_trans.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    return {
        "label_names": list(label_names),
        "log10_labels": sorted([k for k in label_names if k in log10_labels]),
        "d_clamp_min": float(d_clamp_min),
        "mean": mean.astype(np.float64).tolist(),
        "std": std.astype(np.float64).tolist(),
    }


# ==================================================
#  Training and Evaluation
# ==================================================
def train_one_epoch(model, loader, optimizer, scaler, epoch, args, sampler, label_norm=None, label_names=None):
    model.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    mean_torch = None
    std_torch = None
    d_clamp_min = None
    log10_labels = None
    if label_norm is not None and label_names is not None and args.task_type == "reg":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mean_torch = torch.tensor(label_norm["mean"], device=dev, dtype=torch.float32)
        std_torch = torch.tensor(label_norm["std"], device=dev, dtype=torch.float32)
        d_clamp_min = float(label_norm.get("d_clamp_min", MD_D_CLAMP_MIN))
        log10_labels = set(label_norm.get("log10_labels", []))
    
    if getattr(args, "freeze_encoder", False):
        m = model.module if hasattr(model, "module") else model
        m.embed_tokens.eval()
        m.atom_feature.eval()
        m.edge_feature.eval()
        m.encoder.eval()
        m.se3_invariant_kernel.eval()
        m.reg_head.train()
        if hasattr(m, "seg_only_projs"):
            m.seg_only_projs.train()

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
                if getattr(args, "label_index", -1) >= 0:
                    li = int(args.label_index)
                    if label.dim() == 1:
                        label = label.unsqueeze(-1)
                    if li >= label.size(-1):
                        raise ValueError(f"label_index={li} out of range for label dim={label.size(-1)}")
                    label = label[:, li:li + 1]

                if mean_torch is not None and std_torch is not None:
                    label_t = _label_transform_torch(
                        label,
                        label_names,
                        log10_labels=log10_labels,
                        d_clamp_min=d_clamp_min,
                    )
                    label = (label_t.float() - mean_torch) / std_torch

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
    peak_mem_gb = 0.0
    if torch.cuda.is_available():
        peak_mem_gb = float(torch.cuda.max_memory_allocated()) / (1024.0**3)
    return {"loss": epoch_loss, "peak_mem_gb": peak_mem_gb}


@torch.no_grad()
def evaluate(model, loader, args, label_norm=None, label_names=None):
    model.eval()
    
    if args.task_type == "reg":
        all_pred_z = []
        all_label_raw = []
    else:
        conf_matrix = None
        C = args.num_tasks
        conf_matrix = np.zeros((C, C), dtype=np.int64)
        all_preds = []
        all_labels = []

    for batch in loader:
        # Move batch to CUDA
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)

        # Forward pass
        pred = model(batch)
        label = batch["label"].float().cuda()
        if args.task_type == "reg" and getattr(args, "label_index", -1) >= 0:
            li = int(args.label_index)
            if label.dim() == 1:
                label = label.unsqueeze(-1)
            if li >= label.size(-1):
                raise ValueError(f"label_index={li} out of range for label dim={label.size(-1)}")
            label = label[:, li:li + 1]

        if args.task_type == "reg":
            all_pred_z.append(pred.detach().cpu().numpy())
            all_label_raw.append(label.detach().cpu().numpy())
        else:
            pred_cpu = pred.cpu().numpy()
            label_cpu = label.cpu().numpy().flatten()
            pred_cls = pred_cpu.argmax(axis=-1).flatten()
            
            all_preds.append(pred_cls)
            all_labels.append(label_cpu)

            for t, p in zip(label_cpu, pred_cls):
                conf_matrix[int(t), int(p)] += 1

    if args.task_type == "reg":
        if not all_pred_z:
            return {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0, "residuals": None}

        pred_z = np.concatenate(all_pred_z, axis=0)
        label_raw = np.concatenate(all_label_raw, axis=0)

        if label_norm is None or label_names is None:
            label_names = list(getattr(args, "label_names", [])) or [f"y{i}" for i in range(pred_z.shape[1])]
            label_norm = {
                "label_names": list(label_names),
                "log10_labels": [],
                "d_clamp_min": float(MD_D_CLAMP_MIN),
                "mean": [0.0] * pred_z.shape[1],
                "std": [1.0] * pred_z.shape[1],
            }

        log10_labels = set(label_norm.get("log10_labels", []))
        mean = np.asarray(label_norm["mean"], dtype=np.float64).reshape(1, -1)
        std = np.asarray(label_norm["std"], dtype=np.float64).reshape(1, -1)

        # Ground truth in transformed space (D/tau log10)
        label_trans = label_raw.astype(np.float64, copy=True)
        for i, name in enumerate(label_names):
            if name in log10_labels:
                label_trans[:, i] = np.log10(
                    np.clip(label_trans[:, i], a_min=float(label_norm["d_clamp_min"]), a_max=None)
                )

        label_z = (label_trans - mean) / std
        residuals_z = pred_z.astype(np.float64) - label_z

        def _r2(y_t, y_p):
            ss_res = float(np.sum((y_p - y_t) ** 2))
            ss_tot = float(np.sum((y_t - float(np.mean(y_t))) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        z_per_label = {}
        raw_per_label = {}
        for j, name in enumerate(label_names):
            y_t_z = label_z[:, j]
            y_p_z = pred_z[:, j].astype(np.float64)
            z_per_label[name] = {
                "MAE": float(np.mean(np.abs(y_p_z - y_t_z))),
                "RMSE": float(np.sqrt(np.mean((y_p_z - y_t_z) ** 2))),
                "R2": float(_r2(y_t_z, y_p_z)),
            }

        # Raw-space metrics (after inverse transform). Note: D==0 becomes clamp_min by design.
        pred_trans = pred_z.astype(np.float64) * std + mean
        pred_raw_proc = _label_inverse_transform_np(pred_trans, label_names, log10_labels=log10_labels)
        label_raw_proc = _label_inverse_transform_np(label_trans, label_names, log10_labels=log10_labels)

        for j, name in enumerate(label_names):
            y_t = label_raw_proc[:, j]
            y_p = pred_raw_proc[:, j]
            raw_per_label[name] = {
                "MAE": float(np.mean(np.abs(y_p - y_t))),
                "RMSE": float(np.sqrt(np.mean((y_p - y_t) ** 2))),
                "R2": float(_r2(y_t, y_p)),
            }

        mae_z_macro = float(np.mean([m["MAE"] for m in z_per_label.values()]))
        rmse_z_macro = float(np.mean([m["RMSE"] for m in z_per_label.values()]))
        r2_z_macro = float(np.mean([m["R2"] for m in z_per_label.values()]))

        mae_raw_macro = float(np.mean([m["MAE"] for m in raw_per_label.values()]))
        rmse_raw_macro = float(np.mean([m["RMSE"] for m in raw_per_label.values()]))
        r2_raw_macro = float(np.mean([m["R2"] for m in raw_per_label.values()]))

        return {
            "MAE": mae_z_macro,
            "RMSE": rmse_z_macro,
            "R2": r2_z_macro,
            "MAE_raw": mae_raw_macro,
            "RMSE_raw": rmse_raw_macro,
            "R2_raw": r2_raw_macro,
            "z_per_label": z_per_label,
            "raw_per_label": raw_per_label,
            "residuals": residuals_z,
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
    model = MultiMolModel(args)
        
    # ============ Block for finetune task & Directory Setup ============
    if args.main_task in ("finetune", "chain", "chain_only"):
        is_unimol2 = (args.weight_path is not None and "unimol2" in os.path.basename(args.weight_path))
        if (not getattr(args, "no_pretrained", False)) and args.weight_path is not None:
            try:
                model = load_monomer_pretrained_weights(model, args.weight_path, args)
            except FileNotFoundError:
                print("Checkpoint file not found, training from scratch.")

        # Pretrain-only heads are unused in finetune/chain tasks.
        freeze_pretrain_heads(model)

        # Finetune stage-1-only runs: freeze chain-level modules so trainable == optim params.
        if args.main_task == "finetune" and getattr(args, "finetune_optimizer_stage1_only", False):
            m = model.module if hasattr(model, "module") else model
            for module_name in ("chain_token_feature", "chain_edge_feature", "chain_encoder", "seg_only_projs"):
                if hasattr(m, module_name):
                    mod = getattr(m, module_name)
                    for p in mod.parameters():
                        p.requires_grad = False
                    mod.eval()
        if args.freeze_encoder:
            freeze_for_chain_stage1(model)    
        model = model.cuda()
        if args.distributed:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,)
        run_dir = (
            Path(args.results_root)
            / args.dataset_name
            / f"fold_{args.fold}"
            / f"wd_{args.weight_decay}_lr_{args.lr}_do_{args.dropout}_wogroup_{args.wo_pair}"
        )
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

        locked_test_loader = None
        if args.locked_test_pkl_path:
            locked_test_loader, _ = build_dataloader(args.locked_test_pkl_path, args, mode="full")

        label_names = None
        label_norm = None
        if args.task_type == "reg":
            label_names = _resolve_label_names_from_dataset(train_loader.dataset, args)
            args.label_names = label_names  # convenience for debugging
            log10_labels = _parse_log10_labels_arg(getattr(args, "log10_labels", ""))
            label_norm = _compute_label_norm_from_train_dataset(
                train_loader.dataset,
                label_names,
                log10_labels=log10_labels,
                d_clamp_min=float(getattr(args, "d_clamp_min", MD_D_CLAMP_MIN)),
            )
            if is_main_process(args.rank):
                save_json(label_norm, run_dir / "label_norm.json")

    
        # ============ Optimizer & Scheduler ============
        #optimizer = torch.optim.AdamW(
        #    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        #)
        m = model.module if isinstance(model, DDP) else model

        if args.main_task == "finetune" and getattr(args, "finetune_optimizer_stage1_only", False):
            stage1_params = [
                *m.embed_tokens.parameters(),
                *m.atom_feature.parameters(),
                *m.edge_feature.parameters(),
                *m.encoder.parameters(),
                *m.se3_invariant_kernel.parameters(),
                *(m.stage1_capacity_mlp.parameters() if getattr(m, "stage1_capacity_mlp", None) is not None else []),
                *m.reg_head.parameters(),
            ]
            optimizer = torch.optim.AdamW(
                stage1_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        elif args.main_task in ("chain", "chain_only") and args.freeze_encoder:
            optimizer = torch.optim.AdamW(
                [
                    *m.chain_token_feature.parameters(),
                    *m.chain_edge_feature.parameters(),
                    *m.chain_encoder.parameters(),
                    *(m.seg_only_projs.parameters() if hasattr(m, "seg_only_projs") else []),
                    *m.reg_head.parameters(),
                ],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        if is_main_process(args.rank):
            trainable_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optim_numel = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
            print(f"[ParamCount] trainable_numel={trainable_numel} optim_numel={optim_numel}")
            
        scaler = GradScaler(enabled=args.amp)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(10, args.epochs // 3),
            T_mult=1,
            eta_min=args.lr * 0.01,
        )
        # ============ Training Loop ============
        metrics_history = {}
        best_score = float("inf") if args.task_type == "reg" else 0.0
        
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
        
            train_stats = train_one_epoch(
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
            train_loss = float(train_stats["loss"])
            peak_mem_gb = float(train_stats.get("peak_mem_gb", 0.0))
            scheduler.step()
        
            val_metrics = evaluate(model, val_loader, args, label_norm=label_norm, label_names=label_names)
            epoch_time = time.time() - epoch_start
        
            if is_main_process(args.rank):
                # ----- Logging & Printing -----
                if args.task_type == "reg":
                    print(
                        f"[E{epoch:03d}] "
                        f"train_loss={train_loss:.{METRIC_PRINT_DIGITS}f} "
                        f"val_mae={val_metrics['MAE']:.{METRIC_PRINT_DIGITS}f} "
                        f"val_rmse={val_metrics['RMSE']:.{METRIC_PRINT_DIGITS}f} "
                        f"val_r2={val_metrics['R2']:.{METRIC_PRINT_DIGITS}f} "
                        f"peak_mem_gb={peak_mem_gb:.3f} "
                        f"time={epoch_time:.1f}s"
                    )
                else:
                    print(
                        f"[E{epoch:03d}] "
                        f"train_loss={train_loss:.{METRIC_PRINT_DIGITS}f} "
                        f"val_acc={val_metrics['ACC']:.{METRIC_PRINT_DIGITS}f} "
                        f"peak_mem_gb={peak_mem_gb:.3f} "
                        f"time={epoch_time:.1f}s"
                    )
        
                # ----- Save Metrics -----
                metrics_history[epoch] = {
                    "train_loss": train_loss,
                    "peak_mem_gb": peak_mem_gb,
                    **val_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
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
                    best_test_metrics = None
                    if locked_test_loader is not None:
                        test_metrics = evaluate(
                            model,
                            locked_test_loader,
                            args,
                            label_norm=label_norm,
                            label_names=label_names,
                        )
                        best_test_metrics = {
                            "epoch": epoch,
                            **test_metrics,
                        }
                        save_json(best_test_metrics, run_dir / "best_test_metrics.json")

                    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                    save_checkpoint_finetune(
                        {
                            "epoch": epoch,
                            "model_state": model_state,
                            "optimizer_state": optimizer.state_dict(),
                            "scaler_state": scaler.state_dict(),
                            "best_test_metrics": best_test_metrics,
                            "label_norm": label_norm,
                            "args": vars(args),
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
                        f"val_rmse={val_metrics['RMSE']:.{METRIC_PRINT_DIGITS}f} < {args.target_rmse:.{METRIC_PRINT_DIGITS}f} "
                        f"(max_epoch={args.target_rmse_max_epoch})"
                    )
                break
            
            if args.early_stop_patience and epoch >= int(getattr(args, "early_stop_start_epoch", 1)):
                start_epoch = int(getattr(args, "early_stop_start_epoch", 1))
                effective_best_epoch = best_epoch if best_epoch >= start_epoch else (start_epoch - 1)
                if epoch - effective_best_epoch >= args.early_stop_patience:
                    if is_main_process(args.rank):
                        print(
                            f"Early stopping triggered at epoch {epoch}, "
                            f"best was epoch {best_epoch}, "
                            f"effective_best_for_patience was {effective_best_epoch}, "
                            f"best score was {best_score:.{METRIC_PRINT_DIGITS}f}"
                        )
                    break
        
        if args.distributed:
            dist.destroy_process_group()
    
    # ============ Block for pretrain task ============
    if args.main_task == "pretrain":
    
        # -------- Dataloader & model --------
        model = model.cuda()
        train_path = args.pretrain_train_path
        val_path   = args.pretrain_val_path
        pre_loader, pre_sampler = build_dataloader(train_path, args, mode="full")
        val_loader, _           = build_dataloader(val_path,   args, mode="full")
    
        # -------- Directory setup --------
        run_dir  = Path(args.results_root) / "random_pretrain"
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
                model.load_state_dict(ckpt["model_state"], strict=True)
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
