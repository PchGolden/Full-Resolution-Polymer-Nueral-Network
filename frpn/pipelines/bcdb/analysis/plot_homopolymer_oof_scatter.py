#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class Spec:
    model_tag: str
    color: str


LABEL_KEYS = [
    ("density", "density"),
    ("rg", "Rg"),
    ("self_diffusion", "self-diffusion"),
    ("cp", "Cp"),
    ("dielectric_const", "dielectric_const_dc"),
    ("refractive_index", "refractive_index"),
]

FIGSIZE = (4.4, 4.4)
FIG_DPI = 220


def _latest_ckpt_for_fold(root: Path, fold: int) -> Optional[Path]:
    candidates: List[Tuple[float, Path]] = []
    for p in root.rglob("checkpoints/checkpoint.pt"):
        if f"fold_{fold}" not in str(p):
            continue
        try:
            candidates.append((p.stat().st_mtime, p))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _ensure_src_on_path() -> None:
    import sys

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "src"))


def _inverse_transform_np(y_trans: np.ndarray, label_norm: Dict) -> np.ndarray:
    label_names = list(label_norm.get("label_names", []))
    log10_labels = set(label_norm.get("log10_labels", []))
    out = y_trans.copy()
    for i, name in enumerate(label_names):
        if name in log10_labels:
            out[:, i] = np.power(10.0, out[:, i])
    return out


def _predict_one_fold(
    ckpt_path: Path,
    pkl_path: Path,
    fold: int,
    *,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    _ensure_src_on_path()
    import main as bcdb_main  # type: ignore
    from models.multi_mol_model import MultiMolModel  # type: ignore
    from dataloader.dataloader_polymer import build_dataloader  # type: ignore

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    args_dict = ckpt.get("args", {})

    # Rebuild args object using main.parse_args(), then override with ckpt args.
    # This is safer than constructing a raw Namespace since frpn/pipelines/bcdb/main.py expects many fields.
    argv = [
        "prog",
        "--dataset_name",
        "_eval",
        "--main_task",
        str(args_dict.get("main_task", "chain")),
        "--task_type",
        str(args_dict.get("task_type", "reg")),
        "--num_tasks",
        str(args_dict.get("num_tasks", 1)),
        "--fold",
        str(fold),
        "--pkl_path",
        str(pkl_path),
        "--results_root",
        "_",
        "--epochs",
        "1",
        "--batch_size",
        str(batch_size),
        "--lr",
        str(args_dict.get("lr", 1e-4)),
        "--weight_decay",
        str(args_dict.get("weight_decay", 1e-4)),
        "--dropout",
        str(args_dict.get("dropout", 0.1)),
        "--seed",
        str(args_dict.get("seed", 42)),
        "--max_chain_tokens",
        str(args_dict.get("max_chain_tokens", 1536)),
        "--num_workers",
        "0",
    ]

    import sys

    sys.argv = argv
    args = bcdb_main.parse_args()
    args.rank = 0
    args.world_size = 1
    args.distributed = False

    # Carry over relevant booleans from ckpt args.
    for k in [
        "wo_triopm",
        "wo_pair",
        "wo_geom_3d",
        "wo_edge",
        "wo_spd",
        "wo_node",
        "freeze_encoder",
        "equalize_active_params",
        "disable_anchor_symmetry_break",
        "label_zscore",
    ]:
        if k in args_dict:
            setattr(args, k, bool(args_dict[k]))

    if "stage2_only_repr" in args_dict:
        args.stage2_only_repr = str(args_dict["stage2_only_repr"])

    if "log10_labels" in args_dict:
        args.log10_labels = str(args_dict["log10_labels"])
    if "log10_clamp_min" in args_dict:
        args.log10_clamp_min = float(args_dict["log10_clamp_min"])

    bcdb_main.set_seed(int(getattr(args, "seed", 42)))
    bcdb_main.apply_pkl_metadata_to_args(args)

    model = MultiMolModel(args)
    model.load_state_dict(ckpt["model_state"], strict=True)
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    val_loader, _ = build_dataloader(str(pkl_path), args, mode="val")

    label_norm = ckpt.get("label_norm", None)
    label_names = None
    if label_norm is not None:
        label_names = list(label_norm.get("label_names", []))

    y_true_raw_list: List[np.ndarray] = []
    y_pred_raw_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in val_loader:
            for kk, vv in batch.items():
                if torch.is_tensor(vv):
                    batch[kk] = vv.to(dev)
            pred = model(batch)  # could be z-space if label_zscore
            y_raw = batch["label"].float()

            pred_np = pred.detach().cpu().numpy()
            y_np = y_raw.detach().cpu().numpy()

            if label_norm is not None and label_names is not None and bool(args_dict.get("label_zscore", False)):
                mean = np.asarray(label_norm["mean"], dtype=np.float64).reshape(1, -1)
                std = np.asarray(label_norm["std"], dtype=np.float64).reshape(1, -1)
                pred_trans = pred_np.astype(np.float64) * std + mean

                log10_set = set(label_norm.get("log10_labels", []))
                y_trans = y_np.astype(np.float64).copy()
                for i, name in enumerate(label_names):
                    if name in log10_set:
                        y_trans[:, i] = np.log10(np.clip(y_trans[:, i], a_min=float(label_norm.get("log10_clamp_min", 1e-16)), a_max=None))

                pred_raw = _inverse_transform_np(pred_trans, label_norm)
                y_raw_proc = _inverse_transform_np(y_trans, label_norm)

                y_true_raw_list.append(y_raw_proc[:, 0:1])
                y_pred_raw_list.append(pred_raw[:, 0:1])
            else:
                y_true_raw_list.append(y_np[:, 0:1])
                y_pred_raw_list.append(pred_np[:, 0:1])

    y_true = np.concatenate(y_true_raw_list, axis=0).reshape(-1)
    y_pred = np.concatenate(y_pred_raw_list, axis=0).reshape(-1)
    return y_true, y_pred


def _plot_scatter(path: Path, y_true: np.ndarray, y_pred: np.ndarray, *, title: str, color: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(y_true, y_pred, s=10, alpha=0.55, c=color, edgecolors="none")

    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    pad = 0.03 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color="#777777", lw=1.0, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("True", fontsize=9)
    ax.set_ylabel("Pred", fontsize=9)
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_scatter_overlay(
    path: Path, series: List[Tuple[str, np.ndarray, np.ndarray, str]], *, title: str
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)
    ax = fig.add_subplot(1, 1, 1)

    all_true = []
    all_pred = []
    for _, y_true, y_pred, _ in series:
        all_true.append(y_true)
        all_pred.append(y_pred)
    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)

    lo = float(np.nanmin([y_true_all.min(), y_pred_all.min()]))
    hi = float(np.nanmax([y_true_all.max(), y_pred_all.max()]))
    pad = 0.03 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad

    ax.plot([lo, hi], [lo, hi], color="#777777", lw=1.0, zorder=0)

    for name, y_true, y_pred, color in series:
        ax.scatter(y_true, y_pred, s=10, alpha=0.55, c=color, edgecolors="none", label=name)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("True", fontsize=10)
    ax.set_ylabel("Pred", fontsize=10)
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=9, loc="best")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot homopolymer OOF scatter (true vs predicted) per model×label.")
    ap.add_argument("--results_root", default="results_homopolymer_mdstyle/HOMO_372")
    ap.add_argument("--pkl_dir", default="data/processed/HOMO_372/homopolymer/main")
    ap.add_argument("--out_dir", default="paper_results/homopolymer_oof_scatter")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--cache",
        action="store_true",
        help="Cache per-model×label OOF y_true/y_pred to npz to avoid rerunning inference when tweaking plots.",
    )
    ap.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable cache read/write (always rerun inference).",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    pkl_dir = Path(args.pkl_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache_oof_npz"
    cache_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        Spec("frpn", "#CC0000"),  # red
        Spec("stage1_only", "#1F5AA6"),  # blue
        Spec("stage2_only_smiles", "#000000"),  # black
    ]

    summary_rows = []
    missing = []

    for label_key, label_raw in LABEL_KEYS:
        pkl_path = pkl_dir / f"split_{label_key}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(pkl_path)

        overlay_series: List[Tuple[str, np.ndarray, np.ndarray, str]] = []

        for spec in specs:
            cache_path = cache_dir / f"oof__{spec.model_tag}__{label_key}.npz"
            use_cache = bool(args.cache) and not bool(args.no_cache)
            if use_cache and cache_path.exists():
                blob = np.load(cache_path)
                y_true_all = blob["y_true"].astype(np.float64)
                y_pred_all = blob["y_pred"].astype(np.float64)
                n_total = int(y_true_all.size)
            else:
                fold_true = []
                fold_pred = []
                n_total = 0
                for fold in range(int(args.folds)):
                    label_root = (
                        results_root / spec.model_tag / f"label{[k for k,_ in LABEL_KEYS].index(label_key)}_{label_key}"
                    )
                    ckpt = _latest_ckpt_for_fold(label_root, fold)
                    if ckpt is None:
                        missing.append((spec.model_tag, label_key, fold, str(label_root)))
                        continue
                    device = args.device
                    if device == "cuda" and not torch.cuda.is_available():
                        device = "cpu"
                    y_true, y_pred = _predict_one_fold(
                        ckpt,
                        pkl_path,
                        fold,
                        batch_size=int(args.batch_size),
                        device=device,
                    )
                    fold_true.append(y_true)
                    fold_pred.append(y_pred)
                    n_total += y_true.size

                if not fold_true:
                    continue
                y_true_all = np.concatenate(fold_true, axis=0)
                y_pred_all = np.concatenate(fold_pred, axis=0)
                if use_cache:
                    np.savez_compressed(cache_path, y_true=y_true_all.astype(np.float32), y_pred=y_pred_all.astype(np.float32))

            rmse = float(np.sqrt(np.mean((y_pred_all - y_true_all) ** 2)))
            summary_rows.append(
                {
                    "model": spec.model_tag,
                    "label_key": label_key,
                    "label_raw": label_raw,
                    "n_points": int(n_total),
                    "rmse_oof": rmse,
                }
            )

            out_path = out_dir / "scatter" / spec.model_tag / f"{label_key}.png"
            _plot_scatter(
                out_path,
                y_true_all,
                y_pred_all,
                title=f"{label_raw}",
                color=spec.color,
            )

            display_name = {
                "frpn": "FRPN",
                "stage1_only": "Stage1-only",
                "stage2_only_smiles": "Stage2-only",
            }.get(spec.model_tag, spec.model_tag)
            if spec.model_tag in {"frpn", "stage1_only"}:
                overlay_series.append((display_name, y_true_all, y_pred_all, spec.color))

        if overlay_series:
            overlay_path = out_dir / "scatter_overlay" / f"{label_key}.png"
            _plot_scatter_overlay(overlay_path, overlay_series, title=f"{label_raw}")

    (out_dir / "summary.json").write_text(json.dumps({"missing": missing}, indent=2))
    import pandas as pd

    pd.DataFrame(summary_rows).sort_values(["label_key", "model"]).to_csv(out_dir / "summary.csv", index=False)
    print(f"[OK] wrote plots to {out_dir/'scatter'}")
    print(f"[OK] wrote summary to {out_dir/'summary.csv'}")
    if missing:
        print(f"[WARN] missing checkpoints for {len(missing)} model×label×fold entries (see summary.json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
