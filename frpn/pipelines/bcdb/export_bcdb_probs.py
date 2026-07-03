#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export full validation-set probabilities from a saved BCDB checkpoint.

Why this exists:
  - Training-time validation under DDP uses DistributedSampler, so per-epoch
    saved preds/labels are sharded and incomplete on rank0.
  - This script forces single-process evaluation and saves per-sample
    probabilities for downstream ROC/AUC plotting.

Example:
  python frpn/pipelines/bcdb/export_bcdb_probs.py \
    --checkpoint results_xxx/.../checkpoints/checkpoint.pt \
    --pkl_path data/processed/BCDB_chain_withvolume.pkl \
    --fold 0 \
    --output_npz paper_results/extra_ablation_probs_20260522/model/fold_0.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataloader.dataloader_polymer import build_dataloader
from models.multi_mol_model import MultiMolModel


def _load_checkpoint(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint format at {path}")
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint missing key 'model_state': {path}")
    return ckpt


def _namespace_from_ckpt_args(ckpt_args: Dict[str, Any]) -> argparse.Namespace:
    # Best-effort: older checkpoints may miss keys; MultiMolModel uses getattr defaults.
    return argparse.Namespace(**(ckpt_args or {}))


def _override_eval_args(a: argparse.Namespace, cli: argparse.Namespace) -> argparse.Namespace:
    a.distributed = False
    a.world_size = 1
    a.rank = 0
    a.local_rank = 0

    a.pkl_path = cli.pkl_path
    a.fold = int(cli.fold)

    # Export-time dataloader settings
    a.batch_size = int(cli.batch_size)
    a.num_workers = int(cli.num_workers)
    a.pin_memory = bool(cli.pin_memory)

    # Export-time forward settings
    a.amp = bool(cli.amp)

    return a


@torch.no_grad()
def _run_eval(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_idx_all: list[int] = []
    y_true_all: list[int] = []
    prob_pos_all: list[float] = []
    logits_all: list[np.ndarray] = []

    model.eval()

    for batch in loader:
        row_idx = batch.get("row_idx", None)
        if row_idx is None:
            raise KeyError("Batch missing 'row_idx' (expected from preprocessing_polymer.py).")
        row_idx_all.extend([int(x) for x in row_idx])

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        label = batch["label"].squeeze(-1).long()

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(batch)

        if logits.dim() != 2 or logits.size(-1) < 2:
            raise ValueError(f"Expected logits shape [B, C>=2], got {tuple(logits.shape)}")

        prob_pos = F.softmax(logits, dim=-1)[:, 1]

        y_true_all.extend(label.detach().cpu().numpy().astype(np.int64).tolist())
        prob_pos_all.extend(prob_pos.detach().cpu().numpy().astype(np.float64).tolist())
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))

    row_idx_np = np.asarray(row_idx_all, dtype=np.int64)
    y_true_np = np.asarray(y_true_all, dtype=np.int64)
    prob_pos_np = np.asarray(prob_pos_all, dtype=np.float64)
    logits_np = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0, 2), dtype=np.float32)
    return row_idx_np, y_true_np, prob_pos_np, logits_np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export BCDB probabilities from a checkpoint (single-process).")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoints/checkpoint.pt")
    p.add_argument("--pkl_path", required=True, help="Path to data/processed/*.pkl (with folds)")
    p.add_argument("--fold", type=int, required=True, help="Fold index to export (val split uses fold{k})")
    p.add_argument("--eval_split", choices=["val", "full"], default="val")
    p.add_argument("--output_npz", required=True, help="Output .npz path")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--amp", action="store_true")

    args = p.parse_args(argv)

    ckpt = _load_checkpoint(args.checkpoint)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    model_args = _namespace_from_ckpt_args(ckpt_args)
    model_args = _override_eval_args(model_args, args)

    if getattr(model_args, "task_type", None) != "cls":
        raise ValueError(f"Export script expects classification checkpoints (task_type=cls), got {getattr(model_args, 'task_type', None)!r}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = MultiMolModel(model_args)
    # Handle fp_table buffer shape mismatch (stage2_only fp checkpoints).
    fp_table = ckpt["model_state"].get("fp_table", None) if isinstance(ckpt.get("model_state", None), dict) else None
    if isinstance(fp_table, torch.Tensor) and hasattr(model, "fp_table") and torch.is_tensor(getattr(model, "fp_table")):
        if tuple(model.fp_table.shape) != tuple(fp_table.shape):
            model.fp_table = torch.zeros(fp_table.shape, dtype=fp_table.dtype)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")

    model = model.to(device)

    split = "val" if args.eval_split == "val" else "full"
    loader, _ = build_dataloader(args.pkl_path, model_args, mode=split)

    row_idx, y_true, prob_pos, logits = _run_eval(
        model=model,
        loader=loader,
        device=device,
        amp=bool(model_args.amp),
    )

    # Compute metrics (best-effort)
    auc_roc = float("nan")
    acc = float("nan")
    bacc = float("nan")
    f1 = float("nan")
    mcc = float("nan")
    try:
        from sklearn.metrics import (
            roc_auc_score,
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            matthews_corrcoef,
        )  # type: ignore

        # roc_auc_score requires both classes present.
        if len(np.unique(y_true)) >= 2:
            auc_roc = float(roc_auc_score(y_true, prob_pos))
        y_pred = (prob_pos >= 0.5).astype(np.int64)
        acc = float(accuracy_score(y_true, y_pred))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception as e:
        print(f"[METRICS][WARN] Failed to compute some metrics: {e}")

    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "pkl_path": str(Path(args.pkl_path).resolve()),
        "fold": int(args.fold),
        "eval_split": args.eval_split,
        "main_task": getattr(model_args, "main_task", None),
        "task_type": getattr(model_args, "task_type", None),
        "num_tasks": int(getattr(model_args, "num_tasks", 0) or 0),
        "auc_roc": auc_roc,
        "acc": acc,
        "bacc": bacc,
        "f1": f1,
        "mcc": mcc,
        "n_samples": int(len(y_true)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
    }

    np.savez_compressed(
        out_path,
        row_idx=row_idx,
        y_true=y_true,
        prob_pos=prob_pos,
        logits=logits,
        auc_roc=np.asarray(auc_roc, dtype=np.float64),
        acc=np.asarray(acc, dtype=np.float64),
        bacc=np.asarray(bacc, dtype=np.float64),
        f1=np.asarray(f1, dtype=np.float64),
        mcc=np.asarray(mcc, dtype=np.float64),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=str),
    )

    print(
        f"[OK] Saved -> {out_path} (n={len(y_true)} "
        f"auc={auc_roc} acc={acc} bacc={bacc} f1={f1} mcc={mcc} "
        f"ckpt_epoch={meta['checkpoint_epoch']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
