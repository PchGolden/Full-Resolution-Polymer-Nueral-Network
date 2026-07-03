#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Summarize homopolymer runs from training logs.

Input:
  - log files matching:
      HOMO_3x6x5_<jobid>_<task>.out
      HOMO_adapt_2x6x5_<jobid>_<task>.out

Parse:
  - header line: [HOMO] ... / [HOMO-ADAPT] ...
  - epoch lines: [E###] ... val_rmse=... val_r2=...

For each task log:
  - select best epoch by minimum val_rmse
  - keep corresponding val_r2

Aggregate:
  - per model x label: mean/std over folds
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


HDR_RE = re.compile(r"\[(?:HOMO|HOMO-ADAPT)\]\s+idx=\d+\s+fold=(\d+)\s+model=([^\s]+)\s+label=([^\n\r]+)")
EP_ID_RE = re.compile(r"^\[E(\d+)\]\s+")
KV_RE = re.compile(
    r"\b(val_rmse_raw|val_r2_raw|val_rmse|val_r2)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b"
)


def parse_one(path: Path):
    txt = path.read_text(errors="ignore")
    m = HDR_RE.search(txt)
    if not m:
        return None
    fold = int(m.group(1))
    model = m.group(2).strip()
    label = m.group(3).strip()

    epochs = []
    for line in txt.splitlines():
        line = line.strip()
        m_ep = EP_ID_RE.match(line)
        if not m_ep:
            continue
        ep = int(m_ep.group(1))
        kv = {k: float(v) for k, v in KV_RE.findall(line)}
        if "val_rmse" not in kv or "val_r2" not in kv:
            continue
        epochs.append(
            (
                ep,
                float(kv["val_rmse"]),
                float(kv["val_r2"]),
                float(kv.get("val_rmse_raw", kv["val_rmse"])),
                float(kv.get("val_r2_raw", kv["val_r2"])),
            )
        )
    if not epochs:
        return None
    best = min(epochs, key=lambda x: x[1])  # min val_rmse (z-space if normalized)
    return {
        "log_file": str(path),
        "fold": fold,
        "model": model,
        "label": label,
        "best_epoch": best[0],
        "best_rmse": best[1],
        "best_r2": best[2],
        "best_rmse_raw": best[3],
        "best_r2_raw": best[4],
        "n_epochs_logged": len(epochs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--jobid_main", required=True)
    ap.add_argument(
        "--jobid_adapt",
        default="",
        help="Optional jobid for an additional HOMO_adapt array (if provided, those logs are included).",
    )
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    logs += sorted(log_dir.glob(f"HOMO_3x6x5_{args.jobid_main}_*.out"))
    if args.jobid_adapt:
        logs += sorted(log_dir.glob(f"HOMO_adapt_2x6x5_{args.jobid_adapt}_*.out"))

    rows = []
    for p in logs:
        r = parse_one(p)
        if r is not None:
            rows.append(r)

    if not rows:
        raise RuntimeError("No parsable logs found.")

    df = pd.DataFrame(rows)
    per_fold_path = out_dir / "homopolymer_best_by_fold.csv"
    df.sort_values(["model", "label", "fold"]).to_csv(per_fold_path, index=False)

    g = df.groupby(["model", "label"], as_index=False).agg(
        folds=("fold", "count"),
        mean_best_rmse=("best_rmse", "mean"),
        std_best_rmse=("best_rmse", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        mean_best_r2=("best_r2", "mean"),
        std_best_r2=("best_r2", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        mean_best_rmse_raw=("best_rmse_raw", "mean"),
        std_best_rmse_raw=("best_rmse_raw", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        mean_best_r2_raw=("best_r2_raw", "mean"),
        std_best_r2_raw=("best_r2_raw", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
    )
    summary_path = out_dir / "homopolymer_summary_model_label.csv"
    g.sort_values(["model", "label"]).to_csv(summary_path, index=False)

    print(f"[OK] wrote {per_fold_path}")
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
