#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


MODELS = ["Uni_Macro", "Stage2_Only", "FRPN_FP", "FRPN"]
LABELS = ["density", "Rg", "D", "S_q_peak", "nematic_order", "dielectric_constant", "refractive_index"]
LOG10_LABELS = {"D"}
MODEL_DISPLAY = {
    "Uni_Macro": "Uni-Macro",
    "Stage2_Only": "Stage2-Only",
    "FRPN_FP": "FRPN-FP",
    "FRPN": "FRPN",
}
LABEL_DISPLAY = {
    "density": "Density",
    "Rg": "Rg",
    "D": "D",
    "dielectric_constant": "EPS",
    "nematic_order": "N. Order",
    "S_q_peak": "S_peak",
    "refractive_index": "RI",
}
BLUE_NAVY_COLORS = [
    "#C9E0F5",
    "#AECBE2",
    "#7FA8CC",
    "#4F739D",
    "#254468",
    "#0B1B33",
]


@dataclass
class PairMetrics:
    control: str
    label: str
    model: str
    n_pairs: int
    corr_delta: float
    mae_delta: float
    sign_acc: float
    mean_abs_true_delta: float
    mean_abs_pred_delta: float


def _json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _transform_y(y_raw: np.ndarray, label: str) -> np.ndarray:
    y = np.asarray(y_raw, dtype=np.float64)
    if label in LOG10_LABELS:
        y = np.log10(np.clip(y, 1e-30, None))
    return y


def _temp_stats(raw_df: pd.DataFrame, labels: Sequence[str]) -> Dict[Tuple[int, str], Tuple[float, float]]:
    stats: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for temp, sub in raw_df.groupby("glob_feat0", dropna=False):
        t = int(temp)
        for lab in labels:
            vals = _transform_y(sub[lab].to_numpy(), lab)
            mu = float(np.nanmean(vals))
            sd = float(np.nanstd(vals))
            if not np.isfinite(sd) or sd < 1e-12:
                sd = 1.0
            stats[(t, lab)] = (mu, sd)
    return stats


def _zscore_by_temp(y_raw: np.ndarray, temps: np.ndarray, label: str, stats: Dict[Tuple[int, str], Tuple[float, float]]) -> np.ndarray:
    y_t = _transform_y(y_raw, label)
    out = np.empty_like(y_t, dtype=np.float64)
    for temp in np.unique(temps):
        mu, sd = stats[(int(temp), label)]
        m = temps == temp
        out[m] = (y_t[m] - mu) / sd
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or y.size < 3:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(err * err)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, effect_threshold: float = 1.0) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = _rmse(y_true, y_pred)
    corr = _pearson(y_true, y_pred)
    mask = np.abs(y_true) >= float(effect_threshold)
    if int(mask.sum()) > 0:
        valid = np.sign(y_true[mask]) != 0
        sign_acc = float(np.mean((np.sign(y_true[mask][valid]) == np.sign(y_pred[mask][valid])).astype(np.float64))) if int(valid.sum()) > 0 else float("nan")
    else:
        sign_acc = float("nan")
    return {"mae": mae, "rmse": rmse, "corr": corr, "sign_acc": sign_acc}


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _load_predictions(preds_dir: Path, model: str, label: str) -> pd.DataFrame:
    path = preds_dir / f"oof_{model}__{label}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    required = {"row_idx", "y_true_raw", "y_pred_raw", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df[["row_idx", "y_true_raw", "y_pred_raw", "fold"]].copy()


def _load_embedding(embeds_dir: Path, model: str, label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = embeds_dir / f"oof_emb_{model}__{label}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding file: {path}")
    z = np.load(path)
    row_idx = np.asarray(z["row_idx"], dtype=np.int64)
    emb = np.asarray(z["emb"])
    fold = np.asarray(z["fold"], dtype=np.int64)
    if row_idx.ndim != 1 or emb.ndim != 2 or fold.ndim != 1:
        raise ValueError(f"Bad embedding shapes for {path}: row_idx={row_idx.shape}, emb={emb.shape}, fold={fold.shape}")
    if row_idx.shape[0] != emb.shape[0] or row_idx.shape[0] != fold.shape[0]:
        raise ValueError(f"Embedding alignment mismatch for {path}")
    return row_idx, emb, fold


def _build_composition(raw_df: pd.DataFrame) -> Tuple[List[Dict[str, int]], np.ndarray, List[str], List[str]]:
    smiles_cols = [f"SMILES{i}" for i in range(8) if f"SMILES{i}" in raw_df.columns]
    count_cols = [f"seg{i}_feat0" for i in range(8) if f"seg{i}_feat0" in raw_df.columns]
    all_smiles: List[str] = []
    for c in smiles_cols:
        all_smiles.extend([s for s in raw_df[c].dropna().astype(str).tolist() if s and s != "nan"])
    vocab = sorted(set(all_smiles))
    vocab_idx = {s: i for i, s in enumerate(vocab)}

    comp_counts_list: List[Dict[str, int]] = []
    comp_vec = np.zeros((len(raw_df), len(vocab)), dtype=np.float32)
    top1_smiles: List[str] = []

    for ridx, row in raw_df.iterrows():
        counts: Dict[str, int] = {}
        total = 0
        pairs: List[Tuple[int, str]] = []
        for sc, cc in zip(smiles_cols, count_cols):
            s = row.get(sc, None)
            c = row.get(cc, None)
            if not isinstance(s, str) or not s or s == "nan":
                continue
            try:
                ci = int(round(float(c)))
            except Exception:
                continue
            if ci <= 0:
                continue
            counts[s] = counts.get(s, 0) + ci
            total += ci
            pairs.append((ci, s))
        comp_counts_list.append(counts)
        if total > 0:
            for s, ci in counts.items():
                comp_vec[ridx, vocab_idx[s]] = float(ci) / float(total)
        if pairs:
            pairs.sort(reverse=True)
            top1_smiles.append(pairs[0][1])
        else:
            top1_smiles.append("")
    return comp_counts_list, comp_vec, vocab, top1_smiles


def _assign_len_bin(meta: pd.DataFrame, q: int = 4) -> pd.DataFrame:
    meta = meta.copy()
    meta["len_bin"] = "all"
    for temp, sub_idx in meta.groupby("glob_feat0", dropna=False).groups.items():
        idx = list(sub_idx)
        chain_len = meta.loc[idx, "chain_len"].astype(np.int64)
        try:
            bins = pd.qcut(chain_len, q=q, duplicates="drop").astype(str)
            meta.loc[idx, "len_bin"] = bins.values
        except Exception:
            meta.loc[idx, "len_bin"] = "all"
    return meta


def _one_hot(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return pd.get_dummies(df[list(cols)].astype(str), columns=list(cols), drop_first=False)


def _crossfit_predict(X: np.ndarray, y: np.ndarray, fold: np.ndarray, model_ctor) -> np.ndarray:
    yhat = np.full_like(y, np.nan, dtype=np.float64)
    for f in sorted(np.unique(fold)):
        tr = fold != f
        va = fold == f
        mdl = model_ctor()
        mdl.fit(X[tr], y[tr])
        yhat[va] = mdl.predict(X[va])
    return yhat


def _crossfit_linear(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    return _crossfit_predict(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), fold, lambda: LinearRegression())


def _crossfit_ridge(X: np.ndarray, y: np.ndarray, fold: np.ndarray, alpha: float) -> np.ndarray:
    return _crossfit_predict(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), fold, lambda: Ridge(alpha=float(alpha)))


def _plot_heatmap(
    mat: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    title: str,
    *,
    cmap="coolwarm",
    sequential: bool = False,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(9.0, 1.12 * mat.shape[1])
    fig_h = max(4.8, 0.68 * mat.shape[0])
    vals = mat.to_numpy(dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        raise ValueError("heatmap matrix is empty")
    if sequential:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-6
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        vlim = float(np.nanmax(np.abs(finite)))
        if vlim < 1e-12:
            vlim = 1.0
        vmin = -vlim
        vmax = vlim
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        center=None if sequential else 0.0,
        annot=False,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = vals[i, j]
            if not np.isfinite(val):
                continue
            rgba = cmap_obj(norm(val))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = "black" if luminance > 0.62 else "white"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=txt_color,
            )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("label", fontsize=12, labelpad=12)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0, labelsize=12, pad=4)
    ax.tick_params(axis="y", labelsize=12, pad=10)
    xlabels = [LABEL_DISPLAY.get(str(lbl), str(lbl)) for lbl in mat.columns]
    ylabels = [MODEL_DISPLAY.get(str(lbl), str(lbl)) for lbl in mat.index]
    ax.set_xticklabels(xlabels, fontweight="bold", rotation=0, ha="center")
    ax.set_yticklabels(ylabels, fontweight="bold", rotation=0, va="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def _plot_2x2_bars(
    df: pd.DataFrame,
    *,
    value_col: str,
    out_png: Path,
    out_pdf: Path,
    title: str,
    ylabel: str,
    model_order: Sequence[str],
    label_order: Sequence[str],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=220, sharex=True, sharey=True)
    axes = axes.ravel()
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    values = df[value_col].to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"no finite values in {value_col}")
    ymin = float(np.nanmin(finite))
    ymax = float(np.nanmax(finite))
    pad = max(0.003, 0.18 * max(abs(ymin), abs(ymax), 1e-6))
    y0 = min(0.0, ymin - pad)
    y1 = max(0.0, ymax + pad)
    span = max(y1 - y0, 1e-6)
    for ax, model, color in zip(axes, model_order, colors):
        sub = df[df["model"] == model].copy()
        sub["label"] = pd.Categorical(sub["label"], categories=list(label_order), ordered=True)
        sub = sub.sort_values("label")
        vals = sub[value_col].to_numpy(dtype=np.float64)
        bars = ax.bar(sub["label"].astype(str), vals, color=color, alpha=0.9, width=0.54)
        ax.axhline(0.0, color="black", linewidth=1.1, linestyle="--", dashes=(4, 3))
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, bottom=True, left=True, length=4, width=1.0)
        ax.tick_params(axis="x", rotation=0, labelsize=12, pad=4)
        ax.tick_params(axis="y", labelsize=12, pad=4)
        display_labels = [LABEL_DISPLAY.get(str(lbl), str(lbl)) for lbl in sub["label"].astype(str)]
        ax.set_xticks(np.arange(len(display_labels)))
        ax.set_xticklabels(display_labels, fontweight="bold", rotation=0, ha="center")
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.set_ylim(y0, y1)
        for rect, val in zip(bars, vals):
            if not np.isfinite(val):
                continue
            offset = 0.02 * span
            ypos = val + (offset if val >= 0 else -offset)
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                ypos,
                f"{val:+.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )
    for ax in axes[2:]:
        ax.set_xlabel("label")
    for ax in axes[::2]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def _prepare_meta(raw_csv: Path, with_fold_csv: Path, out_meta_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    raw = pd.read_csv(raw_csv).copy()
    raw.insert(0, "row_idx", np.arange(len(raw), dtype=np.int64))
    with_fold = pd.read_csv(with_fold_csv).copy()
    if "row_idx" not in with_fold.columns:
        with_fold.insert(0, "row_idx", np.arange(len(with_fold), dtype=np.int64))
    if len(raw) != len(with_fold):
        raise ValueError(f"raw rows ({len(raw)}) and with_fold rows ({len(with_fold)}) differ")

    if not np.array_equal(raw["row_idx"].to_numpy(dtype=np.int64), with_fold["row_idx"].to_numpy(dtype=np.int64)):
        raise ValueError("raw and with_fold row_idx are not aligned")

    comp_counts_list, comp_vec, vocab, top1_smiles = _build_composition(raw)
    meta = raw[["row_idx", "topology", "mix_mode", "glob_feat0", "rigid_ratio", "polar_ratio", "chem_profile_id"]].copy()
    meta["chain_len"] = raw["chain_node_seg_id"].map(lambda s: len(json.loads(s)) if isinstance(s, str) and s else -1).astype(np.int64)
    meta["len_bin"] = _assign_len_bin(meta, q=4)["len_bin"]
    meta["fold"] = with_fold["fold"].to_numpy(dtype=np.int64)
    meta["top1_smiles"] = top1_smiles
    meta["comp_counts_json"] = [json.dumps(d, sort_keys=True) for d in comp_counts_list]
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_meta_dir / "sample_meta.csv", index=False)
    np.save(out_meta_dir / "comp_vec.npy", comp_vec)
    (out_meta_dir / "comp_vocab.json").write_text(json.dumps(vocab, indent=2))

    return meta, comp_vec, vocab, top1_smiles


def _alignment_qc(
    raw: pd.DataFrame,
    with_fold: pd.DataFrame,
    meta: pd.DataFrame,
    preds_dir: Path,
    embeds_dir: Path,
    out_path: Path,
) -> Dict:
    raw = raw.copy()
    if "row_idx" not in raw.columns:
        raw.insert(0, "row_idx", np.arange(len(raw), dtype=np.int64))
    if "row_idx" not in with_fold.columns:
        with_fold = with_fold.copy()
        with_fold.insert(0, "row_idx", np.arange(len(with_fold), dtype=np.int64))

    structural_cols = ["topology", "glob_feat0", "rigid_ratio", "polar_ratio", "chem_profile_id", "mix_mode"]
    structural_match = {}
    for col in structural_cols:
        if col in raw.columns and col in with_fold.columns:
            ra = raw[col].astype(str).fillna("NA").to_numpy()
            wb = with_fold[col].astype(str).fillna("NA").to_numpy()
            structural_match[col] = bool(np.array_equal(ra, wb))

    pred_rowidx_exact = {}
    pred_fold_exact = {}
    ytrue_diff = {}
    for lab in LABELS:
        diffs = []
        for model in MODELS:
            df = _load_predictions(preds_dir, model, lab)
            pred_rowidx_exact[f"{model}__{lab}"] = bool(np.array_equal(df["row_idx"].to_numpy(dtype=np.int64), np.arange(len(raw), dtype=np.int64)))
            pred_fold_exact[f"{model}__{lab}"] = bool(np.array_equal(df["fold"].to_numpy(dtype=np.int64), with_fold["fold"].to_numpy(dtype=np.int64)))
            diffs.append(np.abs(df["y_true_raw"].to_numpy(dtype=np.float64) - raw[lab].to_numpy(dtype=np.float64)))
        ytrue_diff[lab] = {
            "max_abs_diff": float(np.max(diffs[0])) if diffs else float("nan"),
            "mean_abs_diff": float(np.mean(diffs[0])) if diffs else float("nan"),
        }

    emb_rowidx_exact = {}
    emb_fold_exact = {}
    emb_shapes = {}
    for lab in LABELS:
        for model in MODELS:
            row_idx, emb, fold = _load_embedding(embeds_dir, model, lab)
            emb_rowidx_exact[f"{model}__{lab}"] = bool(np.array_equal(row_idx, np.arange(len(raw), dtype=np.int64)))
            emb_fold_exact[f"{model}__{lab}"] = bool(np.array_equal(fold.astype(np.int64), with_fold["fold"].to_numpy(dtype=np.int64)))
            emb_shapes[f"{model}__{lab}"] = [int(emb.shape[0]), int(emb.shape[1])]

    qc = {
        "raw_rows": int(len(raw)),
        "with_fold_rows": int(len(with_fold)),
        "meta_rows": int(len(meta)),
        "structural_match": structural_match,
        "pred_rowidx_exact": pred_rowidx_exact,
        "pred_fold_exact": pred_fold_exact,
        "emb_rowidx_exact": emb_rowidx_exact,
        "emb_fold_exact": emb_fold_exact,
        "embedding_shapes": emb_shapes,
        "y_true_raw_diff_by_label": ytrue_diff,
        "row_idx_min": int(raw["row_idx"].min()),
        "row_idx_max": int(raw["row_idx"].max()),
    }
    _atomic_write_json(out_path, qc)
    return qc


def _pairwise_overlaps(comp_vec: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    A = comp_vec[idx_a]
    B = comp_vec[idx_b]
    dist = np.abs(A[:, None, :] - B[None, :, :]).sum(axis=-1)
    return 1.0 - 0.5 * dist


def _build_topology_pairs(meta: pd.DataFrame, comp_vec: np.ndarray, threshold: float = 0.95) -> pd.DataFrame:
    rows = []
    group_cols = ["glob_feat0", "mix_mode", "len_bin", "chem_profile_id", "rigid_ratio", "polar_ratio"]
    for key, g in meta.groupby(group_cols, dropna=False):
        if g["topology"].nunique(dropna=False) < 2:
            continue
        by_topology = {str(t): sub["row_idx"].to_numpy(dtype=np.int64) for t, sub in g.groupby("topology", dropna=False)}
        topos = sorted(by_topology.keys())
        for ta, tb in combinations(topos, 2):
            idx_a = by_topology[ta]
            idx_b = by_topology[tb]
            if idx_a.size == 0 or idx_b.size == 0:
                continue
            overlaps = _pairwise_overlaps(comp_vec, idx_a, idx_b)
            ia, ib = np.where(overlaps >= float(threshold))
            for a_i, b_i in zip(ia.tolist(), ib.tolist()):
                ra = int(idx_a[a_i])
                rb = int(idx_b[b_i])
                rows.append(
                    {
                        "match_type": "topology_control",
                        "glob_feat0": float(g["glob_feat0"].iloc[0]),
                        "len_bin": str(g["len_bin"].iloc[0]),
                        "mix_mode": str(g["mix_mode"].iloc[0]),
                        "chem_profile_id": str(g["chem_profile_id"].iloc[0]),
                        "row_i": ra,
                        "row_j": rb,
                        "overlap": float(overlaps[a_i, b_i]),
                        "topology_i": str(meta.loc[meta["row_idx"] == ra, "topology"].iloc[0]),
                        "topology_j": str(meta.loc[meta["row_idx"] == rb, "topology"].iloc[0]),
                        "chain_len_i": int(meta.loc[meta["row_idx"] == ra, "chain_len"].iloc[0]),
                        "chain_len_j": int(meta.loc[meta["row_idx"] == rb, "chain_len"].iloc[0]),
                    }
                )
    return pd.DataFrame(rows)


def _build_chemistry_pairs(meta: pd.DataFrame, comp_vec: np.ndarray, threshold: float = 0.60) -> pd.DataFrame:
    rows = []
    group_cols = ["glob_feat0", "topology", "mix_mode", "len_bin"]
    for key, g in meta.groupby(group_cols, dropna=False):
        idx = g["row_idx"].to_numpy(dtype=np.int64)
        if idx.size < 2:
            continue
        overlaps = _pairwise_overlaps(comp_vec, idx, idx)
        iu, ju = np.triu_indices(idx.size, k=1)
        keep = overlaps[iu, ju] <= float(threshold)
        iu = iu[keep]
        ju = ju[keep]
        for a_i, b_i in zip(iu.tolist(), ju.tolist()):
            ra = int(idx[a_i])
            rb = int(idx[b_i])
            rows.append(
                {
                    "match_type": "chemistry_control",
                    "glob_feat0": float(g["glob_feat0"].iloc[0]),
                    "len_bin": str(g["len_bin"].iloc[0]),
                    "topology": str(g["topology"].iloc[0]),
                    "mix_mode": str(g["mix_mode"].iloc[0]),
                    "row_i": ra,
                    "row_j": rb,
                    "overlap": float(overlaps[a_i, b_i]),
                    "chem_profile_id_i": str(meta.loc[meta["row_idx"] == ra, "chem_profile_id"].iloc[0]),
                    "chem_profile_id_j": str(meta.loc[meta["row_idx"] == rb, "chem_profile_id"].iloc[0]),
                    "chain_len_i": int(meta.loc[meta["row_idx"] == ra, "chain_len"].iloc[0]),
                    "chain_len_j": int(meta.loc[meta["row_idx"] == rb, "chain_len"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def _pair_metrics(
    pairs: pd.DataFrame,
    *,
    raw: pd.DataFrame,
    preds_dir: Path,
    stats: Dict[Tuple[int, str], Tuple[float, float]],
    control_name: str,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    raw = raw.copy()
    if "row_idx" not in raw.columns:
        raw.insert(0, "row_idx", np.arange(len(raw), dtype=np.int64))
    raw_idx = raw.set_index("row_idx")
    row_i = pairs["row_i"].to_numpy(dtype=np.int64)
    row_j = pairs["row_j"].to_numpy(dtype=np.int64)
    temps = pairs["glob_feat0"].to_numpy(dtype=np.int64)

    records: List[PairMetrics] = []
    for lab in LABELS:
        yi = raw_idx.loc[row_i, lab].to_numpy(dtype=np.float64)
        yj = raw_idx.loc[row_j, lab].to_numpy(dtype=np.float64)
        dy = _zscore_by_temp(yi, temps, lab, stats) - _zscore_by_temp(yj, temps, lab, stats)
        for model in MODELS:
            pred = _load_predictions(preds_dir, model, lab)
            pred = pred.merge(raw[["row_idx", "glob_feat0"]], on="row_idx", how="left", validate="one_to_one")
            if pred["glob_feat0"].isna().any():
                raise ValueError(f"Missing glob_feat0 after merge for {model} {lab}")
            yp = pred["y_pred_raw"].to_numpy(dtype=np.float64)
            yi_p = yp[row_i]
            yj_p = yp[row_j]
            dyp = _zscore_by_temp(yi_p, temps, lab, stats) - _zscore_by_temp(yj_p, temps, lab, stats)
            m = _metrics(dy, dyp)
            records.append(
                PairMetrics(
                    control=control_name,
                    label=lab,
                    model=model,
                    n_pairs=int(len(dy)),
                    corr_delta=float(m["corr"]),
                    mae_delta=float(m["mae"]),
                    sign_acc=float(m["sign_acc"]),
                    mean_abs_true_delta=float(np.mean(np.abs(dy))) if len(dy) else float("nan"),
                    mean_abs_pred_delta=float(np.mean(np.abs(dyp))) if len(dy) else float("nan"),
                )
            )
    return pd.DataFrame([r.__dict__ for r in records])


def _pair_metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return metrics.groupby(["control", "model"], as_index=False).agg(
        mean_corr_delta=("corr_delta", "mean"),
        mean_mae_delta=("mae_delta", "mean"),
        mean_sign_acc=("sign_acc", "mean"),
        total_pairs=("n_pairs", "sum"),
    )


def _additive_null_analysis(
    raw: pd.DataFrame,
    meta: pd.DataFrame,
    preds_dir: Path,
    comp_vec: np.ndarray,
    out_dir: Path,
    *,
    chem_k: int = 6,
    ridge_alpha: float = 1.0,
    seed: int = 42,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = raw.copy()
    if "row_idx" not in raw.columns:
        raw.insert(0, "row_idx", np.arange(len(raw), dtype=np.int64))
    fold = meta["fold"].to_numpy(dtype=np.int64)
    temp = raw["glob_feat0"].to_numpy(dtype=np.int64)

    km = KMeans(n_clusters=int(chem_k), random_state=int(seed), n_init=20)
    chem_group = km.fit_predict(comp_vec[meta["row_idx"].to_numpy(dtype=np.int64)]).astype(str)
    meta = meta.copy()
    meta["chem_group"] = chem_group
    X_add = _one_hot(meta, ["chem_group", "topology", "mix_mode", "glob_feat0"]).to_numpy(dtype=np.float64)

    true_resid = {}
    pred_resid = {m: {} for m in MODELS}
    cal_rows = []
    stack_rows = []

    for lab in LABELS:
        y_true_raw = raw[lab].to_numpy(dtype=np.float64)
        y_true_z = _zscore_by_temp(y_true_raw, temp, lab, _temp_stats(raw, [lab]))
        y_null_true = _crossfit_predict(X_add, y_true_z, fold, lambda: LinearRegression())
        r_true = y_true_z - y_null_true
        true_resid[lab] = r_true

        pred_z = {}
        for model in MODELS:
            pred = _load_predictions(preds_dir, model, lab)
            pred = pred.merge(raw[["row_idx", "glob_feat0"]], on="row_idx", how="left", validate="one_to_one")
            y_pred_z = _zscore_by_temp(pred["y_pred_raw"].to_numpy(dtype=np.float64), pred["glob_feat0"].to_numpy(dtype=np.int64), lab, _temp_stats(raw, [lab]))
            pred_z[model] = y_pred_z
            r_m = y_pred_z - _crossfit_predict(X_add, y_pred_z, fold, lambda: LinearRegression())
            pred_resid[model][lab] = r_m

        ens_z = _crossfit_predict(np.stack([pred_z["Uni_Macro"], pred_z["Stage2_Only"]], axis=1), y_true_z, fold, lambda: LinearRegression())
        ens_resid = ens_z - _crossfit_predict(X_add, ens_z, fold, lambda: LinearRegression())
        pred_resid["ensemble_us"] = pred_resid.get("ensemble_us", {})
        pred_resid["ensemble_us"][lab] = ens_resid

        cal_target = r_true
        cal_baseline = ens_resid
        cal_base_rmse = _rmse(cal_target, cal_baseline)
        for model in MODELS:
            cal_pred = _crossfit_predict(pred_resid[model][lab].reshape(-1, 1), cal_target, fold, lambda: LinearRegression())
            rmse = _rmse(cal_target, cal_pred)
            cal_rows.append(
                {
                    "label": lab,
                    "model": model,
                    "cal_rmse": rmse,
                    "ensemble_us_rmse": cal_base_rmse,
                    "cal_rmse_delta_model_minus_ensemble": rmse - cal_base_rmse,
                    "calibrated_better_than_ensemble": bool(rmse < cal_base_rmse),
                }
            )

        X_full = np.stack([pred_resid[m][lab] for m in MODELS], axis=1)
        full_pred = _crossfit_ridge(X_full, cal_target, fold, alpha=float(ridge_alpha))
        full_rmse = _rmse(cal_target, full_pred)
        for drop_model in MODELS:
            keep = [m for m in MODELS if m != drop_model]
            X_drop = np.stack([pred_resid[m][lab] for m in keep], axis=1)
            drop_pred = _crossfit_ridge(X_drop, cal_target, fold, alpha=float(ridge_alpha))
            drop_rmse = _rmse(cal_target, drop_pred)
            stack_rows.append(
                {
                    "label": lab,
                    "model": drop_model,
                    "stack_rmse_full4": full_rmse,
                    "stack_rmse_without_model": drop_rmse,
                    "leave_one_out_gain": drop_rmse - full_rmse,
                    "leave_one_out_hurts_without_model": bool(drop_rmse > full_rmse),
                }
            )

    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(out_dir / "calibrated_residual_metrics_by_label.csv", index=False)
    cal_summary = cal_df.groupby("model", as_index=False).agg(
        mean_cal_rmse=("cal_rmse", "mean"),
        mean_ensemble_us_rmse=("ensemble_us_rmse", "mean"),
        mean_delta=("cal_rmse_delta_model_minus_ensemble", "mean"),
        n_labels=("label", "nunique"),
    )
    cal_summary.to_csv(out_dir / "calibrated_residual_metrics_summary.csv", index=False)

    stack_df = pd.DataFrame(stack_rows)
    stack_df.to_csv(out_dir / "leave_one_out_stack_metrics_by_label.csv", index=False)
    stack_summary = stack_df.groupby("model", as_index=False).agg(
        mean_full_rmse=("stack_rmse_full4", "mean"),
        mean_drop_rmse=("stack_rmse_without_model", "mean"),
        mean_gain=("leave_one_out_gain", "mean"),
        n_labels=("label", "nunique"),
    )
    stack_summary.to_csv(out_dir / "leave_one_out_stack_metrics_summary.csv", index=False)

    model_order = MODELS
    label_order = LABELS
    _plot_2x2_bars(
        cal_df,
        value_col="cal_rmse_delta_model_minus_ensemble",
        out_png=out_dir / "additive_null_panel_a_2x2.png",
        out_pdf=out_dir / "additive_null_panel_a_2x2.pdf",
        title="Additive-null calibrated residual RMSE delta vs Uni-Macro + Stage2-Only ensemble",
        ylabel="cal_rmse(model) - cal_rmse(ensemble_us)",
        model_order=model_order,
        label_order=label_order,
    )
    _plot_2x2_bars(
        stack_df,
        value_col="leave_one_out_gain",
        out_png=out_dir / "additive_null_panel_b_2x2.png",
        out_pdf=out_dir / "additive_null_panel_b_2x2.pdf",
        title="Additive-null leave-one-out gain on the 4-model residual stack",
        ylabel="rmse(without model) - rmse(full stack)",
        model_order=model_order,
        label_order=label_order,
    )

    summary = {
        "chem_k": int(chem_k),
        "ridge_alpha": float(ridge_alpha),
        "models": MODELS,
        "labels": LABELS,
        "calibrated_summary": cal_summary.to_dict(orient="records"),
        "stack_summary": stack_summary.to_dict(orient="records"),
    }
    _atomic_write_json(out_dir / "additive_null_summary.json", summary)
    return summary


def _write_tsne_launcher(
    out_dir: Path,
    *,
    raw_csv: Path,
    preds_dir: Path,
    embeds_dir: Path,
    tsne_out_dir: Path,
    perplexity: float,
    max_iter: int,
    seed: int,
    cpus_per_task: int = 4,
    mem: str = "12G",
    time: str = "08:00:00",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / "submit_md1640_tsne_array.sh"
    repo_root = Path(__file__).resolve().parents[2]
    models = " ".join(MODELS)
    labels = " ".join(LABELS)
    script = f"""#!/usr/bin/env bash
#SBATCH --job-name=md1640_tsne
#SBATCH --array=0-{len(MODELS) * len(LABELS) - 1}
#SBATCH --cpus-per-task={int(cpus_per_task)}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={str((out_dir / "slurm_tsne_%A_%a.out"))}
#SBATCH --error={str((out_dir / "slurm_tsne_%A_%a.err"))}

set -euo pipefail

cd {repo_root}

MODELS=({models})
LABELS=({labels})
TASK=$SLURM_ARRAY_TASK_ID
MODEL_INDEX=$((TASK / {len(LABELS)}))
LABEL_INDEX=$((TASK % {len(LABELS)}))
MODEL=${{MODELS[$MODEL_INDEX]}}
LABEL=${{LABELS[$LABEL_INDEX]}}

python {repo_root / "scripts/figures/recompute_tsne_coords_and_plot_by_pred.py"} \
  --raw_csv {raw_csv} \
  --embeds_dir {embeds_dir} \
  --preds_dir {preds_dir} \
  --model \"$MODEL\" \
  --label \"$LABEL\" \
  --out_dir {tsne_out_dir} \
  --value y_pred_raw \
  --cmap blue_navy \
  --perplexity {float(perplexity)} \
  --max_iter {int(max_iter)} \
  --seed {int(seed)}
"""
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", required=True)
    ap.add_argument("--with_fold_csv", required=True)
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--embeds_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--chem_k", type=int, default=6)
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--effect_threshold", type=float, default=1.0)
    ap.add_argument("--topology_overlap", type=float, default=0.95)
    ap.add_argument("--chemistry_overlap", type=float, default=0.60)
    ap.add_argument("--tsne_perplexity", type=float, default=10.0)
    ap.add_argument("--tsne_max_iter", type=int, default=2000)
    ap.add_argument("--tsne_seed", type=int, default=42)
    ap.add_argument("--write_tsne_launcher", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    meta_dir = out_dir / "meta"
    matched_dir = out_dir / "matched_pair_delta"
    additive_dir = out_dir / "additive_null_residual"
    tsne_dir = out_dir / "embedding_tsne_pred_colored"
    meta_dir.mkdir(parents=True, exist_ok=True)
    matched_dir.mkdir(parents=True, exist_ok=True)
    additive_dir.mkdir(parents=True, exist_ok=True)
    tsne_dir.mkdir(parents=True, exist_ok=True)

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    models = [s.strip() for s in args.models.split(",") if s.strip()]
    if labels != LABELS:
        raise ValueError(f"This implementation expects labels {LABELS}; got {labels}")
    if models != MODELS:
        raise ValueError(f"This implementation expects models {MODELS}; got {models}")

    raw = pd.read_csv(args.raw_csv).copy()
    raw.insert(0, "row_idx", np.arange(len(raw), dtype=np.int64))
    with_fold = pd.read_csv(args.with_fold_csv).copy()
    if "row_idx" not in with_fold.columns:
        with_fold.insert(0, "row_idx", np.arange(len(with_fold), dtype=np.int64))
    if not np.array_equal(raw["row_idx"].to_numpy(dtype=np.int64), with_fold["row_idx"].to_numpy(dtype=np.int64)):
        raise ValueError("raw and with_fold row_idx are not aligned")

    meta, comp_vec, vocab, top1_smiles = _prepare_meta(Path(args.raw_csv), Path(args.with_fold_csv), meta_dir)
    stats = _temp_stats(raw, LABELS)

    alignment_qc = _alignment_qc(raw, with_fold, meta, Path(args.preds_dir), Path(args.embeds_dir), meta_dir / "alignment_qc.json")

    topology_pairs = _build_topology_pairs(meta, comp_vec, threshold=float(args.topology_overlap))
    chemistry_pairs = _build_chemistry_pairs(meta, comp_vec, threshold=float(args.chemistry_overlap))
    topology_pairs.to_csv(matched_dir / "topology_control_pairs.csv", index=False)
    chemistry_pairs.to_csv(matched_dir / "chemistry_control_pairs.csv", index=False)

    topology_metrics = _pair_metrics(topology_pairs, raw=raw, preds_dir=Path(args.preds_dir), stats=stats, control_name="topology_control")
    chemistry_metrics = _pair_metrics(chemistry_pairs, raw=raw, preds_dir=Path(args.preds_dir), stats=stats, control_name="chemistry_control")
    topology_metrics.to_csv(matched_dir / "topology_control_delta_metrics_by_label.csv", index=False)
    chemistry_metrics.to_csv(matched_dir / "chemistry_control_delta_metrics_by_label.csv", index=False)
    topology_summary = _pair_metric_summary(topology_metrics)
    chemistry_summary = _pair_metric_summary(chemistry_metrics)
    topology_summary.to_csv(matched_dir / "topology_control_delta_metrics_summary.csv", index=False)
    chemistry_summary.to_csv(matched_dir / "chemistry_control_delta_metrics_summary.csv", index=False)

    _plot_heatmap(
        topology_metrics.pivot_table(index="model", columns="label", values="corr_delta", aggfunc="mean").reindex(index=MODELS, columns=LABELS),
        matched_dir / "topology_control_corr_4models.png",
        matched_dir / "topology_control_corr_4models.pdf",
        title="Topology control: corr(Δŷ_z, Δy_z)",
        cmap=LinearSegmentedColormap.from_list(
            "topology_blue",
            ["#DCEBFA", "#AECBE2", "#7FA8CC", "#4F739D", "#254468", "#0B1B33"],
            N=256,
        ),
        sequential=True,
    )
    _plot_heatmap(
        chemistry_metrics.pivot_table(index="model", columns="label", values="corr_delta", aggfunc="mean").reindex(index=MODELS, columns=LABELS),
        matched_dir / "chemistry_control_corr_4models.png",
        matched_dir / "chemistry_control_corr_4models.pdf",
        title="Chemistry control: corr(Δŷ_z, Δy_z)",
        cmap=LinearSegmentedColormap.from_list(
            "chemistry_red",
            ["#FBE5D9", "#F6C7B3", "#F3A37C", "#EB7856", "#D9473B", "#B6121D"],
            N=256,
        ),
        sequential=True,
    )

    additive_summary = _additive_null_analysis(
        raw=raw,
        meta=meta,
        preds_dir=Path(args.preds_dir),
        comp_vec=comp_vec,
        out_dir=additive_dir,
        chem_k=int(args.chem_k),
        ridge_alpha=float(args.ridge_alpha),
        seed=int(args.tsne_seed),
    )

    launcher = _write_tsne_launcher(
        tsne_dir,
        raw_csv=Path(args.raw_csv),
        preds_dir=Path(args.preds_dir),
        embeds_dir=Path(args.embeds_dir),
        tsne_out_dir=tsne_dir,
        perplexity=float(args.tsne_perplexity),
        max_iter=int(args.tsne_max_iter),
        seed=int(args.tsne_seed),
    )

    summary = {
        "raw_csv": str(Path(args.raw_csv)),
        "with_fold_csv": str(Path(args.with_fold_csv)),
        "preds_dir": str(Path(args.preds_dir)),
        "embeds_dir": str(Path(args.embeds_dir)),
        "models": MODELS,
        "labels": LABELS,
        "meta_dir": str(meta_dir),
        "matched_pair_dir": str(matched_dir),
        "additive_null_dir": str(additive_dir),
        "tsne_dir": str(tsne_dir),
        "tsne_launcher": str(launcher),
        "alignment_qc": alignment_qc,
        "topology_pairs": int(len(topology_pairs)),
        "chemistry_pairs": int(len(chemistry_pairs)),
        "additive_null": additive_summary,
    }
    _atomic_write_json(out_dir / "mechanism_reproduction_summary.json", summary)
    print(f"[DONE] wrote outputs to {out_dir}")
    print(f"[DONE] tsne launcher: {launcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
