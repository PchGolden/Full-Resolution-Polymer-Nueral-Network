#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Recolor existing t-SNE coordinates (already computed) by a continuous value:
  - OOF prediction (default): y_pred_z or y_pred_raw
  - or y_true_z / y_true_raw

This is a plotting-only step. It does NOT recompute t-SNE coordinates, so it
doesn't need GPU.

Expected coord files (from V3 analysis):
  figures/tsne_coords__emb__<LABEL>__<MODEL>.csv
    columns include: row_idx, tsne_x, tsne_y, topology, mix_mode, glob_feat0, ...

Expected OOF preds:
  preds_oof/oof_<MODEL>__<LABEL>.csv
    columns: row_idx, y_true_raw, y_pred_raw, fold
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402


BLUE_NAVY_COLORS = [
    "#C9E0F5",
    "#AECBE2",
    "#7FA8CC",
    "#4F739D",
    "#254468",
    "#0B1B33",
]


LOG10_LABELS = {"D", "MSD_A2", "eta0_Pa_s", "tau_maxwell_s", "Cp", "K_T"}


def _transform(y: np.ndarray, label: str) -> np.ndarray:
    y = y.astype(np.float64)
    if label in LOG10_LABELS:
        y = np.log10(np.clip(y, 1e-30, None))
    return y


def _compute_temp_stats(raw_df: pd.DataFrame, labels: Iterable[str]) -> Dict[Tuple[int, str], Tuple[float, float]]:
    stats: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for temp, sub in raw_df.groupby("glob_feat0"):
        t = int(temp)
        for lab in labels:
            y = _transform(sub[lab].to_numpy(), lab)
            stats[(t, lab)] = (float(np.mean(y)), float(np.std(y) + 1e-12))
    return stats


def _zscore(y_raw: np.ndarray, temp: np.ndarray, label: str, stats: Dict[Tuple[int, str], Tuple[float, float]]) -> np.ndarray:
    y_t = _transform(y_raw, label)
    out = np.empty_like(y_t, dtype=np.float64)
    for t in np.unique(temp):
        mean, std = stats[(int(t), label)]
        m = temp == t
        out[m] = (y_t[m] - mean) / std
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", required=True, help="raw CSV with true labels + glob_feat0")
    ap.add_argument("--coords_csv", required=True, help="tsne_coords__emb__<LABEL>__<MODEL>.csv")
    ap.add_argument("--preds_csv", required=True, help="oof_<MODEL>__<LABEL>.csv")
    ap.add_argument("--label", required=True)
    ap.add_argument("--value", default="y_pred_z", choices=["y_pred_raw", "y_true_raw", "y_pred_z", "y_true_z"])
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--cmap", default="blue_navy")
    ap.add_argument("--vmin_p", type=float, default=2.0, help="lower percentile for color clipping")
    ap.add_argument("--vmax_p", type=float, default=98.0, help="upper percentile for color clipping")
    ap.add_argument("--s", type=float, default=8.0, help="marker size")
    ap.add_argument("--alpha", type=float, default=0.85)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw_csv)
    if "row_idx" not in raw.columns:
        raw.insert(0, "row_idx", np.arange(len(raw), dtype=np.int64))

    coords = pd.read_csv(args.coords_csv)
    preds = pd.read_csv(args.preds_csv)
    label = args.label

    # attach temps for z-score (use raw table)
    preds = preds.merge(raw[["row_idx", "glob_feat0", label]], on="row_idx", how="left", validate="one_to_one")
    if preds["glob_feat0"].isna().any():
        raise ValueError("some preds rows missing glob_feat0 after merge; row_idx mismatch?")

    stats = _compute_temp_stats(raw, [label])

    if args.value in ("y_pred_z", "y_true_z"):
        if args.value == "y_pred_z":
            val = _zscore(preds["y_pred_raw"].to_numpy(), preds["glob_feat0"].to_numpy().astype(int), label, stats)
        else:
            val = _zscore(preds["y_true_raw"].to_numpy(), preds["glob_feat0"].to_numpy().astype(int), label, stats)
        preds = preds.assign(value=val)
    elif args.value == "y_pred_raw":
        preds = preds.assign(value=preds["y_pred_raw"].to_numpy())
    else:
        preds = preds.assign(value=preds["y_true_raw"].to_numpy())

    df = coords.merge(preds[["row_idx", "value"]], on="row_idx", how="inner")
    if df.empty:
        raise RuntimeError("no rows after merging coords and preds (row_idx mismatch)")

    vmin = float(np.percentile(df["value"].to_numpy(), args.vmin_p))
    vmax = float(np.percentile(df["value"].to_numpy(), args.vmax_p))

    if args.cmap == "blue_navy":
        cmap = LinearSegmentedColormap.from_list("blue_navy", BLUE_NAVY_COLORS, N=256)
    else:
        cmap = args.cmap

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.2))
    sc = ax.scatter(
        df["tsne_x"].to_numpy(),
        df["tsne_y"].to_numpy(),
        c=df["value"].to_numpy(),
        cmap=cmap,
        s=args.s,
        alpha=args.alpha,
        linewidths=0,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ttl = args.title.strip()
    if not ttl:
        ttl = f"t-SNE colored by {args.value} ({label})"
    ax.set_title(ttl, fontsize=11)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(args.value, rotation=90)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=300)
    out_pdf = str(Path(args.out_png).with_suffix(".pdf"))
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
