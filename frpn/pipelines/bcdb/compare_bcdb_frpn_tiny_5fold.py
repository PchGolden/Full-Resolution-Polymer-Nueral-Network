#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FoldSpec:
    fold: int
    checkpoint: str


def _latest_ckpt_under_root(root: Path, fold: int) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for p in root.rglob("checkpoint.pt"):
        if f"fold_{fold}" not in str(p):
            continue
        try:
            candidates.append((p.stat().st_mtime, str(p)))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _run_export(export_py: str, ckpt: str, pkl: str, fold: int, out_npz: str, device: str, batch_size: int) -> None:
    cmd = [
        "python",
        export_py,
        "--checkpoint",
        ckpt,
        "--pkl_path",
        pkl,
        "--fold",
        str(fold),
        "--eval_split",
        "val",
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--output_npz",
        out_npz,
    ]
    subprocess.check_call(cmd)


def _read_meta(npz_path: Path) -> Dict:
    with np.load(npz_path, allow_pickle=True) as z:
        meta_json = str(z["meta_json"].tolist())
    return json.loads(meta_json)


def _write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export per-fold probs/metrics and summarize BCDB FRPN-Tiny 5-fold.")
    p.add_argument("--pkl_path", default="data/processed/BCDB_chain_withvolume.pkl")
    p.add_argument("--results_root", default="results_bcdb_frpn_tiny_withvol_40ep/BCDB")
    p.add_argument("--output_root", default="paper_results/bcdb_frpn_tiny_40ep")
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args(argv)

    export_py = str(Path(__file__).with_name("export_bcdb_probs.py"))
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results_root = Path(args.results_root)

    summary_rows: List[Dict[str, object]] = []
    missing_ckpts: List[Dict[str, object]] = []

    for fold in range(int(args.folds)):
        ckpt = _latest_ckpt_under_root(results_root, fold)
        if ckpt is None:
            missing_ckpts.append({"fold": fold, "reason": "missing_checkpoint"})
            continue

        out_npz = out_root / "frpn_tiny" / f"fold_{fold}.npz"
        _run_export(
            export_py=export_py,
            ckpt=ckpt,
            pkl=args.pkl_path,
            fold=fold,
            out_npz=str(out_npz),
            device=args.device,
            batch_size=int(args.batch_size),
        )

        meta = _read_meta(out_npz)
        summary_rows.append(
            {
                "model": "frpn_tiny",
                "fold": fold,
                "checkpoint_epoch": meta.get("checkpoint_epoch", ""),
                "auc_roc": meta.get("auc_roc", ""),
                "acc": meta.get("acc", ""),
                "bacc": meta.get("bacc", ""),
                "f1": meta.get("f1", ""),
                "mcc": meta.get("mcc", ""),
                "checkpoint": meta.get("checkpoint", ""),
            }
        )

    cols = ["model", "fold", "checkpoint_epoch", "auc_roc", "acc", "bacc", "f1", "mcc", "checkpoint"]
    by_fold_path = out_root / "summary_by_fold.csv"
    _write_csv(by_fold_path, summary_rows, cols)

    meanstd_rows: List[Dict[str, object]] = []
    if summary_rows:
        for key in ["auc_roc", "acc", "bacc", "f1", "mcc"]:
            vals = []
            for r in summary_rows:
                try:
                    vals.append(float(r[key]))
                except Exception:
                    pass
            v = np.asarray(vals, dtype=np.float64)
            meanstd_rows.append(
                {
                    "model": "frpn_tiny",
                    "metric": key,
                    "mean": float(np.nanmean(v)) if v.size else float("nan"),
                    "std": float(np.nanstd(v, ddof=1)) if v.size > 1 else float("nan"),
                    "n_folds": int(v.size),
                }
            )

    meanstd_path = out_root / "summary_meanstd.csv"
    _write_csv(meanstd_path, meanstd_rows, ["model", "metric", "mean", "std", "n_folds"])

    report = {
        "results_root": str(results_root),
        "pkl_path": str(Path(args.pkl_path).resolve()),
        "output_root": str(out_root),
        "device": args.device,
        "batch_size": int(args.batch_size),
        "folds": int(args.folds),
        "missing_ckpts": missing_ckpts,
        "summary_by_fold": str(by_fold_path),
        "summary_meanstd": str(meanstd_path),
    }
    report_path = out_root / "summary_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"[OK] Wrote summaries under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
