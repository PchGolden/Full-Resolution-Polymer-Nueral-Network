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
class ModelSpec:
    name: str
    ckpt_roots: List[str]
    fixed_ckpt_pattern: Optional[str] = None


def _latest_ckpt_under_roots(roots: List[str], fold: int) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for r in roots:
        root = Path(r)
        if not root.exists():
            continue
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
    p = argparse.ArgumentParser(description="Export per-fold probs/metrics and summarize (BCDB 5-fold).")
    p.add_argument("--pkl_path", default="data/processed/BCDB_chain_withvolume.pkl")
    p.add_argument("--output_root", default="paper_results/extra_ablation_metrics_20260523")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument(
        "--stage2_fp_root",
        default="results_stage2_fp_morgan2048_withvol/BCDB",
        help="Root dir containing stage2 fp runs (will pick latest checkpoint.pt per fold).",
    )
    p.add_argument(
        "--frpn_ptfreeze_root",
        default="results_frpn_ptfreeze_stage1_withvol/BCDB",
        help="Root dir containing frpn ptfreeze runs (will pick latest checkpoint.pt per fold).",
    )
    p.add_argument(
        "--frpn_full_ckpt_pattern",
        default=(
            "results/BCDB/task_chain_eq_0_freeze_0/"
            "data_BCDB_chain_withvolume/wt_checkpoint_adaptive/"
            "fold_{fold}/wd_0.01_lr_0.0001_do_0.1_wogroup_False/checkpoints/checkpoint.pt"
        ),
        help="Format string for FRPN(fullchain) checkpoint path per fold.",
    )
    p.add_argument(
        "--frpn_fullchain_adaptive_root",
        default="results_frpn_fullchain_pt_withvol_100ep/BCDB",
        help="Root dir containing FRPN fullchain runs with adaptive init (will pick latest checkpoint.pt per fold).",
    )
    p.add_argument("--folds", type=int, default=5)

    args = p.parse_args(argv)

    export_py = str(Path(__file__).with_name("export_bcdb_probs.py"))
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    models = [
        ModelSpec(name="stage2_fp_morgan2048", ckpt_roots=[args.stage2_fp_root]),
        ModelSpec(name="frpn_ptfreeze_stage1", ckpt_roots=[args.frpn_ptfreeze_root]),
        ModelSpec(name="frpn_fullchain_withvol", ckpt_roots=[], fixed_ckpt_pattern=args.frpn_full_ckpt_pattern),
        ModelSpec(name="frpn_fullchain_adaptive_init", ckpt_roots=[args.frpn_fullchain_adaptive_root]),
    ]

    summary_rows: List[Dict[str, object]] = []

    for fold in range(int(args.folds)):
        for ms in models:
            if ms.fixed_ckpt_pattern is not None:
                ckpt = ms.fixed_ckpt_pattern.format(fold=fold)
                if not Path(ckpt).exists():
                    raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
            else:
                ckpt = _latest_ckpt_under_roots(ms.ckpt_roots, fold)
                if ckpt is None:
                    raise FileNotFoundError(f"Missing checkpoint under roots={ms.ckpt_roots} for fold={fold}")

            out_npz = out_root / ms.name / f"fold_{fold}.npz"
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
                    "model": ms.name,
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
    _write_csv(out_root / "summary_by_fold.csv", summary_rows, cols)

    # mean/std across folds
    meanstd_rows: List[Dict[str, object]] = []
    for ms in models:
        rows = [r for r in summary_rows if r["model"] == ms.name]
        def get_float(key: str) -> np.ndarray:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[key]))
                except Exception:
                    pass
            return np.asarray(vals, dtype=np.float64)

        for key in ["auc_roc", "acc", "bacc", "f1", "mcc"]:
            v = get_float(key)
            meanstd_rows.append(
                {
                    "model": ms.name,
                    "metric": key,
                    "mean": float(np.nanmean(v)) if v.size else float("nan"),
                    "std": float(np.nanstd(v, ddof=1)) if v.size > 1 else float("nan"),
                    "n_folds": int(v.size),
                }
            )

    _write_csv(out_root / "summary_meanstd.csv", meanstd_rows, ["model", "metric", "mean", "std", "n_folds"])

    print(f"[OK] Wrote summaries under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
