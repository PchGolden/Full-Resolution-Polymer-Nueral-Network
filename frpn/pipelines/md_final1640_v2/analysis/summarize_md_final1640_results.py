#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize MD_FINAL1640_V2 single-label regression runs.

Outputs:
  - summary_by_fold.csv
  - summary_meanstd.csv
  - summary_report.json
  - merged OOF embedding npz files (best-effort, from per-fold shards)

This script reads the best epoch from each fold's metrics.json by minimizing
the validation RMSE in z-space, then records the corresponding raw-space RMSE
and R2 values for paper-ready reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _split_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _latest_json(base: Path, name: str) -> Optional[Path]:
    candidates = list(base.rglob(name))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_metrics_path(results_root: Path, dataset_name: str, model: str, label_index: int, label: str, fold: int) -> Optional[Path]:
    base = results_root / dataset_name / model / f"label{label_index}_{label}" / f"fold_{fold}"
    if not base.exists():
        return None
    return _latest_json(base, "metrics.json")


def _find_ckpt_path(results_root: Path, dataset_name: str, model: str, label_index: int, label: str, fold: int) -> Optional[Path]:
    base = results_root / dataset_name / model / f"label{label_index}_{label}" / f"fold_{fold}"
    if not base.exists():
        return None
    ckpts = list(base.rglob("checkpoints/checkpoint.pt"))
    if not ckpts:
        return None
    return sorted(ckpts, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _load_metrics(metrics_path: Path) -> Tuple[int, Dict]:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    epochs = []
    for k in metrics.keys():
        try:
            epochs.append(int(k))
        except Exception:
            continue
    if not epochs:
        raise ValueError(f"metrics.json has no epoch keys: {metrics_path}")
    best_epoch = min(epochs, key=lambda e: float(metrics[str(e)]["RMSE"]))
    return best_epoch, metrics[str(best_epoch)]


def _merge_oof_embeddings(embeds_dir: Path, model: str, label: str) -> Tuple[Optional[Path], int, List[int]]:
    shard_paths = [embeds_dir / model / label / f"fold_{k}.npz" for k in range(5)]
    missing = [k for k, p in enumerate(shard_paths) if not p.exists()]
    if missing:
        return None, 0, missing

    row_idx_all = []
    emb_all = []
    fold_all = []
    for k, p in enumerate(shard_paths):
        z = np.load(str(p))
        row_idx = np.asarray(z["row_idx"], dtype=np.int64)
        emb = np.asarray(z["emb"])
        if row_idx.ndim != 1 or emb.ndim != 2 or row_idx.shape[0] != emb.shape[0]:
            raise ValueError(f"bad embedding shard: {p} row_idx={row_idx.shape} emb={emb.shape}")
        row_idx_all.append(row_idx)
        emb_all.append(emb)
        fold_all.append(np.full((row_idx.shape[0],), k, dtype=np.int8))

    row_idx = np.concatenate(row_idx_all, axis=0)
    emb = np.concatenate(emb_all, axis=0)
    fold = np.concatenate(fold_all, axis=0)

    order = np.argsort(row_idx)
    row_idx = row_idx[order]
    emb = emb[order]
    fold = fold[order]
    uniq, first = np.unique(row_idx, return_index=True)
    if uniq.size != row_idx.size:
        row_idx = row_idx[first]
        emb = emb[first]
        fold = fold[first]

    merged_path = embeds_dir / f"oof_emb_{model}__{label}.npz"
    np.savez_compressed(str(merged_path), row_idx=row_idx, emb=emb, fold=fold)
    return merged_path, int(row_idx.size), []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--dataset_name", default="MD_FINAL1640_V2")
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--embeds_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pkl_path", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    preds_dir = Path(args.preds_dir)
    embeds_dir = Path(args.embeds_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = _split_csv_list(args.models)
    labels = _split_csv_list(args.labels)

    import pickle

    header = pickle.load(open(args.pkl_path, "rb"))
    pkl_label_names = list(header.get("label_names", []))
    name_to_index = {str(n): int(i) for i, n in enumerate(pkl_label_names)}

    by_fold_rows = []
    missing_runs = []
    missing_preds = []
    missing_emb_shards = []
    merged_emb_paths = {}

    for model in models:
        for label in labels:
            if label not in name_to_index:
                missing_runs.append({"model": model, "label": label, "reason": "label_not_in_pkl"})
                continue
            label_index = int(name_to_index[label])

            merged_path, emb_rows, missing_shards = _merge_oof_embeddings(embeds_dir, model, label)
            if missing_shards:
                missing_emb_shards.append({"model": model, "label": label, "missing_folds": missing_shards})
            if merged_path is not None:
                merged_emb_paths[f"{model}__{label}"] = str(merged_path)

            pred_path = preds_dir / f"oof_{model}__{label}.csv"
            if not pred_path.exists():
                missing_preds.append({"model": model, "label": label, "path": str(pred_path)})

            for fold in range(5):
                metrics_path = _find_metrics_path(results_root, args.dataset_name, model, label_index, label, fold)
                ckpt_path = _find_ckpt_path(results_root, args.dataset_name, model, label_index, label, fold)
                if metrics_path is None or ckpt_path is None:
                    missing_runs.append(
                        {
                            "model": model,
                            "label": label,
                            "label_index": label_index,
                            "fold": fold,
                            "metrics_path": str(metrics_path) if metrics_path else None,
                            "ckpt_path": str(ckpt_path) if ckpt_path else None,
                            "reason": "missing_metrics_or_ckpt",
                        }
                    )
                    continue

                best_epoch, best_metrics = _load_metrics(metrics_path)
                by_fold_rows.append(
                    {
                        "dataset_name": args.dataset_name,
                        "model": model,
                        "label_index": label_index,
                        "label": label,
                        "fold": fold,
                        "best_epoch": int(best_epoch),
                        "best_val_rmse_z": float(best_metrics.get("RMSE", np.nan)),
                        "best_val_r2_z": float(best_metrics.get("R2", np.nan)),
                        "best_val_rmse_raw": float(best_metrics.get("RMSE_raw", best_metrics.get("RMSE", np.nan))),
                        "best_val_r2_raw": float(best_metrics.get("R2_raw", best_metrics.get("R2", np.nan))),
                        "best_val_mae_raw": float(best_metrics.get("MAE_raw", best_metrics.get("MAE", np.nan))),
                        "metrics_path": str(metrics_path),
                        "ckpt_path": str(ckpt_path),
                        "pred_path": str(pred_path) if pred_path.exists() else None,
                        "emb_path": str(merged_path) if merged_path is not None else None,
                    }
                )

    by_fold = pd.DataFrame(by_fold_rows)
    by_fold_path = out_dir / "summary_by_fold.csv"
    by_fold.to_csv(by_fold_path, index=False)

    meanstd_rows = []
    if not by_fold.empty:
        group_cols = ["dataset_name", "model", "label_index", "label"]
        for key, sub in by_fold.groupby(group_cols, dropna=False):
            dataset_name, model, label_index, label = key
            meanstd_rows.append(
                {
                    "dataset_name": dataset_name,
                    "model": model,
                    "label_index": int(label_index),
                    "label": label,
                    "n_folds": int(len(sub)),
                    "best_mean_rmse_raw": float(sub["best_val_rmse_raw"].mean()),
                    "best_std_rmse_raw": float(sub["best_val_rmse_raw"].std(ddof=0)),
                    "best_mean_r2_raw": float(sub["best_val_r2_raw"].mean()),
                    "best_std_r2_raw": float(sub["best_val_r2_raw"].std(ddof=0)),
                    "best_mean_rmse_z": float(sub["best_val_rmse_z"].mean()),
                    "best_std_rmse_z": float(sub["best_val_rmse_z"].std(ddof=0)),
                    "best_mean_r2_z": float(sub["best_val_r2_z"].mean()),
                    "best_std_r2_z": float(sub["best_val_r2_z"].std(ddof=0)),
                }
            )

    meanstd = pd.DataFrame(meanstd_rows).sort_values(["model", "label_index"]).reset_index(drop=True)
    meanstd_path = out_dir / "summary_meanstd.csv"
    meanstd.to_csv(meanstd_path, index=False)

    report = {
        "results_root": str(results_root),
        "dataset_name": args.dataset_name,
        "preds_dir": str(preds_dir),
        "embeds_dir": str(embeds_dir),
        "models": models,
        "labels": labels,
        "missing_runs": missing_runs,
        "missing_preds": missing_preds,
        "missing_emb_shards": missing_emb_shards,
        "merged_emb_paths": merged_emb_paths,
        "summary_by_fold": str(by_fold_path),
        "summary_meanstd": str(meanstd_path),
    }
    report_path = out_dir / "summary_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"[OK] wrote {by_fold_path}")
    print(f"[OK] wrote {meanstd_path}")
    print(f"[OK] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
