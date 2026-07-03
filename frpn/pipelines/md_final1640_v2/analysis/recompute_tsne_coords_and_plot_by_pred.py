#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Recompute embedding-space t-SNE coordinates from cached OOF embeddings, then plot colored by
OOF prediction values (or true values).

This does NOT need GPU (sklearn TSNE runs on CPU), but can be submitted to a GPU partition
if desired for scheduling reasons.

Inputs:
  - merged OOF embedding: embeds_dir/oof_emb_<MODEL>__<LABEL>.npz
  - OOF prediction cache: preds_dir/oof_<MODEL>__<LABEL>.csv
  - raw_csv: used for glob_feat0 and label transform/z-scoring when value==*_z
  - optional subset_meta_csv: restrict to a subset row_idx list (e.g., matched subset)

Outputs:
  - out_dir/tsne_coords__emb__<LABEL>__<MODEL>__p<PERP>.csv
  - out_dir/tsne_pred__<LABEL>__<MODEL>__p<PERP>__<VALUE>.png
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


def _load_embedding_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(str(path))
    row_idx = np.asarray(z["row_idx"], dtype=np.int64)
    emb = np.asarray(z["emb"])
    if row_idx.ndim != 1 or emb.ndim != 2 or row_idx.shape[0] != emb.shape[0]:
        raise ValueError(f"bad embedding npz shapes: row_idx={row_idx.shape} emb={emb.shape}")
    return row_idx, emb


def _tsne_2d(x: np.ndarray, perplexity: float, seed: int, max_iter: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    # sklearn changed argument name from n_iter -> max_iter in newer versions.
    # Use a compatibility shim.
    kwargs = dict(
        n_components=2,
        perplexity=float(perplexity),
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
    )
    try:
        return TSNE(**kwargs, max_iter=int(max_iter)).fit_transform(x)
    except TypeError:
        return TSNE(**kwargs, n_iter=int(max_iter)).fit_transform(x)


def _resolve_cmap(name: str, colors: str = ""):
    if colors.strip():
        hex_colors = [c.strip() for c in colors.split(",") if c.strip()]
        return LinearSegmentedColormap.from_list("custom_hex", hex_colors, N=256)
    if name == "blue_navy":
        return LinearSegmentedColormap.from_list("blue_navy", BLUE_NAVY_COLORS, N=256)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", required=True)
    ap.add_argument("--embeds_dir", required=True)
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--subset_meta_csv", default="", help="optional: CSV containing row_idx to restrict plotting")
    ap.add_argument("--perplexity", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--pca_dims", type=int, default=50)
    ap.add_argument("--value", default="y_pred_z", choices=["y_pred_raw", "y_true_raw", "y_pred_z", "y_true_z"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cmap", default="blue_navy", help="matplotlib cmap name (ignored if --colors is set)")
    ap.add_argument(
        "--colors",
        default="",
        help="comma-separated hex colors for a custom linear colormap (low->high), e.g. '#a,#b,#c'",
    )
    ap.add_argument("--vmin_p", type=float, default=2.0)
    ap.add_argument("--vmax_p", type=float, default=98.0)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=0.85)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw_csv)
    if "row_idx" not in raw.columns:
        raw.insert(0, "row_idx", np.arange(len(raw), dtype=np.int64))

    emb_path = Path(args.embeds_dir) / f"oof_emb_{args.model}__{args.label}.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"missing embedding cache: {emb_path}")
    row_idx, emb = _load_embedding_npz(emb_path)

    # optional subset restriction
    if args.subset_meta_csv:
        subset = pd.read_csv(args.subset_meta_csv)
        if "row_idx" not in subset.columns:
            raise ValueError("subset_meta_csv must include row_idx column")
        keep = set(subset["row_idx"].astype(int).tolist())
        mask = np.array([int(r) in keep for r in row_idx], dtype=bool)
        row_idx = row_idx[mask]
        emb = emb[mask]

    # load coords meta (topology/mix_mode/temp) from raw
    meta = raw.set_index("row_idx").loc[row_idx, ["topology", "mix_mode", "glob_feat0"]].reset_index()

    # PCA before TSNE
    emb = emb.astype(np.float32, copy=False)
    from sklearn.decomposition import PCA

    n_comp = int(min(args.pca_dims, emb.shape[1], max(2, emb.shape[0] - 1)))
    if n_comp < emb.shape[1]:
        emb = PCA(n_components=n_comp, random_state=int(args.seed)).fit_transform(emb)

    coords = _tsne_2d(emb, perplexity=float(args.perplexity), seed=int(args.seed), max_iter=int(args.max_iter))
    coords_df = meta.copy()
    coords_df["tsne_x"] = coords[:, 0]
    coords_df["tsne_y"] = coords[:, 1]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_tag = f"p{str(args.perplexity).replace('.', '_')}"

    coords_csv = out_dir / f"tsne_coords__emb__{args.label}__{args.model}__{p_tag}.csv"
    coords_df.to_csv(coords_csv, index=False)

    # prepare coloring values from OOF preds
    preds_path = Path(args.preds_dir) / f"oof_{args.model}__{args.label}.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"missing preds cache: {preds_path}")
    preds = pd.read_csv(preds_path)
    preds = preds.merge(raw[["row_idx", "glob_feat0", args.label]], on="row_idx", how="left", validate="one_to_one")
    if preds["glob_feat0"].isna().any():
        raise ValueError("some preds rows missing glob_feat0 after merge; row_idx mismatch?")

    stats = _compute_temp_stats(raw, [args.label])
    if args.value in ("y_pred_z", "y_true_z"):
        if args.value == "y_pred_z":
            val = _zscore(preds["y_pred_raw"].to_numpy(), preds["glob_feat0"].to_numpy().astype(int), args.label, stats)
        else:
            val = _zscore(preds["y_true_raw"].to_numpy(), preds["glob_feat0"].to_numpy().astype(int), args.label, stats)
        preds = preds.assign(value=val)
    elif args.value == "y_pred_raw":
        preds = preds.assign(value=preds["y_pred_raw"].to_numpy())
    else:
        preds = preds.assign(value=preds["y_true_raw"].to_numpy())

    plot_df = coords_df.merge(preds[["row_idx", "value"]], on="row_idx", how="inner")
    if plot_df.empty:
        raise RuntimeError("no rows after merging coords and preds (row_idx mismatch)")

    vmin = float(np.percentile(plot_df["value"].to_numpy(), args.vmin_p))
    vmax = float(np.percentile(plot_df["value"].to_numpy(), args.vmax_p))

    cmap = _resolve_cmap(args.cmap, args.colors)

    out_png = out_dir / f"tsne_pred__{args.label}__{args.model}__{p_tag}__{args.value}.png"
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.2))
    sc = ax.scatter(
        plot_df["tsne_x"].to_numpy(),
        plot_df["tsne_y"].to_numpy(),
        c=plot_df["value"].to_numpy(),
        cmap=cmap,
        s=float(args.s),
        alpha=float(args.alpha),
        linewidths=0,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    # Title requirement: pure label only.
    ax.set_title(f"{args.label}", fontsize=12)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(args.value, rotation=90)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    out_pdf = out_dir / f"tsne_pred__{args.label}__{args.model}__{p_tag}__{args.value}.pdf"
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)

    print(f"[OK] wrote coords: {coords_csv}")
    print(f"[OK] wrote figure: {out_png}")
    print(f"[OK] wrote figure: {out_pdf}")


if __name__ == "__main__":
    main()
