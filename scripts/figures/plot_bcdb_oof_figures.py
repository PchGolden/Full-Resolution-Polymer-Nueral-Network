#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot 4 required figures from strict 5-fold OOF cache")
    p.add_argument("--meta_csv", default="paper_figures/figure_ver3/cache_fullfold_parentonly_fold1_meta_v1.csv")
    p.add_argument(
        "--cache6_npz",
        default="paper_figures/current_version/cache_6models_oof_5fold_v1.npz",
        help="Unified 6-model OOF cache built by build_cache_6models_oof_5fold.py",
    )
    p.add_argument("--out_dir", default="paper_figures/current_version")
    p.add_argument("--recompute_lda_scores", action="store_true")
    p.add_argument(
        "--model_layout",
        default="2x3",
        choices=["2x3", "1x6"],
        help="Layout for model-panel figures when there are 6 models.",
    )
    p.add_argument(
        "--kimmig_log_glob",
        default="Kimmig_NN/sbatch/sbatch_log/KIMMIG_BCDB_5F_1263373_*.out",
        help="Glob for Kimmig training logs to summarize best-epoch metrics.",
    )
    return p.parse_args()


def _fold_id_from_split(s: str) -> int:
    x = str(s).strip()
    if x.startswith("fold"):
        return int(x.replace("fold", ""))
    return int(x)


def _load_oof_data(args: argparse.Namespace):
    meta_df = pd.read_csv(args.meta_csv)
    y = meta_df["label"].astype(np.int64).values
    fold_id = meta_df["split"].astype(str).map(_fold_id_from_split).astype(np.int64).values
    n = len(meta_df)

    z = np.load(args.cache6_npz)
    expect = np.arange(n, dtype=np.int64)
    if not np.array_equal(np.asarray(z["row_idx"], dtype=np.int64), expect):
        raise RuntimeError("cache6 row_idx is not 0..N-1")
    if not np.array_equal(np.asarray(z["label"], dtype=np.int64), y.astype(np.int64)):
        raise RuntimeError("cache6 label mismatch vs meta_csv")
    if not np.array_equal(np.asarray(z["fold_id"], dtype=np.int64), fold_id.astype(np.int64)):
        raise RuntimeError("cache6 fold_id mismatch vs meta_csv")

    data = {
        "y": y,
        "fold_id": fold_id,
        "probs": {
            "periogt": np.asarray(z["periogt_prob"], dtype=np.float64),
            "transpolymer": np.asarray(z["transpolymer_prob"], dtype=np.float64),
            "sagcn": np.asarray(z["sagcn_prob"], dtype=np.float64),
            "unimacro": np.asarray(z["unimacro_prob"], dtype=np.float64),
            "stage2only": np.asarray(z["stage2only_prob"], dtype=np.float64),
            "frpn": np.asarray(z["frpn_prob"], dtype=np.float64),
        },
        "embs": {
            "periogt": np.asarray(z["periogt_emb"], dtype=np.float64),
            "transpolymer": np.asarray(z["transpolymer_emb"], dtype=np.float64),
            "sagcn": np.asarray(z["sagcn_emb"], dtype=np.float64),
            "unimacro": np.asarray(z["unimacro_emb"], dtype=np.float64),
            "stage2only": np.asarray(z["stage2only_emb"], dtype=np.float64),
            "frpn": np.asarray(z["frpn_emb"], dtype=np.float64),
        },
    }

    # uniform threshold strategy for all models
    data["preds"] = {k: (np.asarray(v, dtype=np.float64) >= 0.5).astype(np.int64) for k, v in data["probs"].items()}
    return data


def _compute_per_model_oof_lda(emb: np.ndarray, y: np.ndarray, fold_id: np.ndarray) -> np.ndarray:
    scores = np.full(y.shape[0], np.nan, dtype=np.float64)
    for f in range(5):
        tr = fold_id != f
        va = fold_id == f
        xtr = emb[tr]
        ytr = y[tr]
        xva = emb[va]

        lda = LinearDiscriminantAnalysis()
        lda.fit(xtr, ytr)

        sc_tr = np.asarray(lda.decision_function(xtr), dtype=np.float64).reshape(-1)
        sc_va = np.asarray(lda.decision_function(xva), dtype=np.float64).reshape(-1)

        mu = float(sc_tr.mean())
        sd = float(sc_tr.std())
        if sd < 1e-12:
            sd = 1.0

        sc_tr = (sc_tr - mu) / sd
        sc_va = (sc_va - mu) / sd

        # enforce direction: positive class should have larger mean score on train set
        pos_mean = float(sc_tr[ytr == 1].mean()) if np.any(ytr == 1) else 0.0
        neg_mean = float(sc_tr[ytr == 0].mean()) if np.any(ytr == 0) else 0.0
        if pos_mean < neg_mean:
            sc_va = -sc_va

        scores[va] = sc_va

    if np.isnan(scores).any():
        raise RuntimeError("NaN remains in computed OOF LDA scores")
    return scores


def _prepare_lda_scores(args: argparse.Namespace, data: dict) -> dict[str, np.ndarray]:
    """Compute OOF LDA scores from embeddings for all models.

    Note: These scores are used ONLY for the LDA histogram/separability view.
    All ROC/AUC metrics are computed from the OOF probabilities in data['probs'].
    """

    out_dir = Path(args.out_dir)
    cache_path = out_dir / "cache_6models_oof_lda_scores_v1.npz"

    y = data["y"]
    fold_id = data["fold_id"]
    specs = _model_specs()

    use_cache = cache_path.exists() and (not args.recompute_lda_scores)
    if use_cache:
        z = np.load(cache_path)
        scores: dict[str, np.ndarray] = {}
        for k, _ in specs:
            scores[k] = np.asarray(z[f"{k}_oof_lda_score"], dtype=np.float64)
        return scores

    scores = {}
    for k, _ in specs:
        scores[k] = _compute_per_model_oof_lda(np.asarray(data["embs"][k], dtype=np.float64), y, fold_id)

    payload = {
        "row_idx": np.arange(len(y), dtype=np.int64),
        "label": y.astype(np.int64),
        "fold_id": fold_id.astype(np.int64),
    }
    for k, _ in specs:
        payload[f"{k}_oof_lda_score"] = np.asarray(scores[k], dtype=np.float32)
    np.savez_compressed(cache_path, **payload)
    return scores


def _model_specs() -> list[tuple[str, str]]:
    # required left-to-right order
    return [
        ("periogt", "PerioGT"),
        ("transpolymer", "TransPolymer"),
        ("sagcn", "SA-GCN"),
        ("unimacro", "Uni-Macro"),
        ("stage2only", "Stage2-Only"),
        ("frpn", "FRPN"),
    ]


def _model_color_map(out_dir: Path) -> dict[str, tuple[float, float, float]]:
    # Mixed palette with added red family from the latest uploaded reference.
    _ = out_dir
    return {
        "periogt": (16 / 255, 70 / 255, 128 / 255),
        "transpolymer": (109 / 255, 173 / 255, 209 / 255),
        # Color adjustment (order matters):
        # 1) swap SA-GCN <-> Uni-Macro
        # 2) swap Uni-Macro <-> Stage2-Only
        # 3) set Stage2-Only to a light red
        "sagcn": (222 / 255, 234 / 255, 234 / 255),  # was Uni-Macro
        "unimacro": (247 / 255, 228 / 255, 116 / 255),  # was Stage2-Only
        "stage2only": (235 / 255, 112 / 255, 63 / 255),  # orange-red
        "frpn": (183 / 255, 34 / 255, 48 / 255),       
    }


def _model_panel_grid(n_models: int, model_layout: str) -> tuple[int, int]:
    if n_models == 6 and model_layout == "2x3":
        return 2, 3
    return 1, n_models


def _set_common_style() -> None:
    sns.set_style("white")
    plt.rcParams["axes.grid"] = False

    # show ticks on all four sides
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True

    # make top/right spines visible
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True

    # ticks point inward on all sides
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    # optional: major tick lengths
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["ytick.major.size"] = 4


def _cohens_d(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    n_pos = int(x_pos.size)
    n_neg = int(x_neg.size)
    if n_pos < 2 or n_neg < 2:
        return np.nan
    v_pos = float(np.var(x_pos, ddof=1))
    v_neg = float(np.var(x_neg, ddof=1))
    pooled = ((n_pos - 1) * v_pos + (n_neg - 1) * v_neg) / float(n_pos + n_neg - 2)
    if pooled <= 0.0:
        return np.nan
    return float((np.mean(x_pos) - np.mean(x_neg)) / np.sqrt(pooled))


def _mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _fold_d_mean_std(scores: np.ndarray, y: np.ndarray, fold_id: np.ndarray) -> tuple[float, float]:
    ds: list[float] = []
    for f in range(5):
        m = fold_id == f
        yf = y[m]
        sf = scores[m]
        d = _cohens_d(sf[yf == 1], sf[yf == 0])
        if np.isfinite(d):
            ds.append(float(d))
    return _mean_std(ds)


def _fold_auc_mean_std(scores: np.ndarray, y: np.ndarray, fold_id: np.ndarray) -> tuple[float, float]:
    aucs: list[float] = []
    for f in range(5):
        m = fold_id == f
        yf = y[m]
        sf = scores[m]
        if np.unique(yf).size == 2:
            aucs.append(float(roc_auc_score(yf, sf)))
    return _mean_std(aucs)


def _write_model_metrics_summary(data: dict, out_dir: Path) -> None:
    y = data["y"]
    fold_id = data["fold_id"]
    rows: list[dict[str, float | str]] = []

    for key, name in _model_specs():
        prob = np.asarray(data["probs"][key], dtype=np.float64)
        pred = np.asarray(data["preds"][key], dtype=np.int64)

        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]

        fold_acc: list[float] = []
        fold_bacc: list[float] = []
        fold_f1: list[float] = []
        fold_mcc: list[float] = []
        fold_auc: list[float] = []
        for f in range(5):
            m = fold_id == f
            yf = y[m]
            pf = pred[m]
            prf = prob[m]
            fold_acc.append(float(accuracy_score(yf, pf)))
            fold_bacc.append(float(balanced_accuracy_score(yf, pf)))
            fold_f1.append(float(f1_score(yf, pf)))
            fold_mcc.append(float(matthews_corrcoef(yf, pf)))
            if np.unique(yf).size == 2:
                fold_auc.append(float(roc_auc_score(yf, prf)))

        row = {
            "model": key,
            "name": name,
            "acc": float(accuracy_score(y, pred)),
            "acc_fold_mean": _mean_std(fold_acc)[0],
            "acc_fold_std": _mean_std(fold_acc)[1],
            "bacc": float(balanced_accuracy_score(y, pred)),
            "bacc_fold_mean": _mean_std(fold_bacc)[0],
            "bacc_fold_std": _mean_std(fold_bacc)[1],
            "f1": float(f1_score(y, pred)),
            "f1_fold_mean": _mean_std(fold_f1)[0],
            "f1_fold_std": _mean_std(fold_f1)[1],
            "mcc": float(matthews_corrcoef(y, pred)),
            "mcc_fold_mean": _mean_std(fold_mcc)[0],
            "mcc_fold_std": _mean_std(fold_mcc)[1],
            "auc": float(roc_auc_score(y, prob)),
            "auc_fold_mean": _mean_std(fold_auc)[0],
            "auc_fold_std": _mean_std(fold_auc)[1],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fig0_model_metrics_summary.csv", index=False)

    print("\n[6-model strict OOF metrics: overall]")
    print(df[["name", "acc", "bacc", "f1", "mcc", "auc"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("[OK] wrote", out_dir / "fig0_model_metrics_summary.csv")


def plot_confusion(data: dict, out_dir: Path, model_layout: str) -> None:
    y = data["y"]
    preds = data["preds"]
    specs = _model_specs()

    nrows, ncols = _model_panel_grid(len(specs), model_layout)

    # Use a ~2:1 canvas ratio for document placement.
    # Keep the layout driven by `--model_layout` (default: 2x3).
    if (nrows, ncols) == (2, 3):
        fig_w, fig_h = 12.0, 6.0
    else:
        fig_w, fig_h = 18.0, 4.5

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes_f = np.ravel(axes)

    for idx, ((k, name), ax) in enumerate(zip(specs, axes_f)):
        cm = confusion_matrix(y, preds[k], labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            annot_kws={"size": 14},
            cmap="Blues",
            cbar=False,
            square=True,
            ax=ax,
            xticklabels=["F", "T"],
            yticklabels=["F", "T"],
        )
        ax.set_title(name, fontsize=10, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("True" if (idx % ncols == 0) else "")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(False)

    # Preserve intended canvas ratio (avoid tight bbox changing dimensions)
    fig.savefig(out_dir / "fig1_confusion_all5fold_concat.png", dpi=1200)
    plt.close(fig)


def plot_roc(data: dict, out_dir: Path) -> None:
    y = data["y"]
    fold_id = data["fold_id"]
    probs = data["probs"]
    color_map = _model_color_map(out_dir)
    specs = _model_specs()

    auc_by_model: dict[str, list[float]] = {k: [] for k, _ in specs}

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), constrained_layout=True)
    for f, ax in enumerate(axes):
        m = fold_id == f
        for k, name in specs:
            fpr, tpr, _ = roc_curve(y[m], probs[k][m])
            auc = float(roc_auc_score(y[m], probs[k][m]))
            auc_by_model[k].append(auc)
            ax.plot(fpr, tpr, color=color_map[k], lw=2.1, label=f"{name} ({auc:.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", lw=1.2)
        ax.set_title(f"Fold{f}", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(False)
        ax.set_box_aspect(1)
        ax.tick_params(top=True, right=True, labeltop=False, labelright=False)
        ax.legend(loc="lower right", fontsize=12, frameon=False)

    fig.savefig(out_dir / "fig2_roc_5fold_by_model.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)

    print("\n[ROC AUC: 5-fold mean ± std]")
    for k, name in specs:
        auc_mean, auc_std = _mean_std(auc_by_model[k])
        print(f"{name}: {auc_mean:.3f} ± {auc_std:.3f}")


def plot_lda(data: dict, lda_scores: dict[str, np.ndarray], out_dir: Path, model_layout: str) -> None:
    y = data["y"]
    fold_id = data["fold_id"]
    specs = _model_specs()

    def _panel_bins_and_xlim(s: np.ndarray) -> tuple[np.ndarray, float, float]:
        s = np.asarray(s, dtype=np.float64)
        lo = float(np.percentile(s, 1.0))
        hi = float(np.percentile(s, 99.0))
        if hi <= lo:
            lo = float(np.percentile(s, 5.0))
            hi = float(np.percentile(s, 95.0))
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.08 * (hi - lo)
        if pad <= 0:
            pad = 1.0
        lo = lo - pad
        hi = hi + pad
        return np.linspace(lo, hi, 40), lo, hi

    nrows, ncols = _model_panel_grid(len(specs), model_layout)
    fig_w = 4.4 * ncols
    fig_h = 4.8 * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes_f = np.ravel(axes)

    for ax, (k, name) in zip(axes_f, specs):
        s = lda_scores[k]
        s0 = s[y == 0]
        s1 = s[y == 1]

        bins, x_lo, x_hi = _panel_bins_and_xlim(s)

        ax.hist(s0, bins=bins, density=False, histtype="step", lw=1.8, color="0.55", label="Neg (F)", zorder=1)
        ax.hist(s1, bins=bins, density=False, histtype="step", lw=2.0, color="k", label="Pos (T)", zorder=2)
        ax.axvline(0.0, color="0.3", lw=1.0, linestyle="--")

        d_mean, d_std = _fold_d_mean_std(s, y, fold_id)
        txt_d_mean = f"{d_mean:.3f}" if np.isfinite(d_mean) else "nan"
        txt_d_std = f"{d_std:.3f}" if np.isfinite(d_std) else "nan"
        ax.text(
            0.03,
            0.97,
            f"Cohen's d={txt_d_mean} ± {txt_d_std}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

        ax.set_title(name, fontsize=12)
        ax.set_xlabel("OOF LDA score (z)")
        ax.set_ylabel("Count")
        ax.set_xlim(x_lo, x_hi)
        ax.grid(False)
        ax.set_box_aspect(1)
        ax.tick_params(top=True, right=True, labeltop=False, labelright=False)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.97, 1.00),
            fontsize=7,
            handlelength=1.2,
            handletextpad=0.4,
            borderpad=0.2,
            labelspacing=0.25,
            frameon=False,
        )

    for j in range(len(specs), len(axes_f)):
        axes_f[j].axis("off")

    fig.savefig(out_dir / "fig3_lda_oof_5fold.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


def plot_other4_disagree(data: dict, out_dir: Path) -> None:
    y = data["y"]
    fold_id = data["fold_id"]
    preds = data["preds"]
    specs = _model_specs()
    color_map = _model_color_map(out_dir)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), constrained_layout=True)

    model_keys = [k for k, _ in specs]
    model_names = [name for _, name in specs]
    model_colors = [color_map[k] for k in model_keys]

    # Per-model disagree-set definition (6 models):
    # For model i, its disagree set is rows where the OTHER five models are not unanimous.
    # This matches "模型在其它五个模型有分歧的样本上的预测准确率".
    disagree_acc_by_model: dict[str, list[float]] = {k: [] for k in model_keys}
    disagree_n_by_model: dict[str, list[int]] = {k: [] for k in model_keys}

    for f, ax in enumerate(axes):
        m = fold_id == f
        yp = y[m]
        pred_mat = np.stack([preds[k][m] for k in model_keys], axis=0)  # (n_models, n)

        model_acc: list[float] = []
        for i, k in enumerate(model_keys):
            others = np.delete(pred_mat, i, axis=0)
            disagree_mask = np.any(others != others[0:1, :], axis=0)
            n_dis = int(disagree_mask.sum())
            disagree_n_by_model[k].append(n_dis)
            if n_dis == 0:
                acc = float("nan")
            else:
                acc = float((pred_mat[i, disagree_mask] == yp[disagree_mask]).mean())
            disagree_acc_by_model[k].append(acc)
            model_acc.append(acc)

        x = np.arange(len(model_keys))
        bars = ax.bar(x, model_acc, color=model_colors, alpha=0.95, width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Fold{f}", fontsize=12)
        ax.set_xlabel("Model")
        ax.set_ylabel("Acc")
        ax.grid(False)
        ax.set_box_aspect(1)
        ax.tick_params(top=True, right=True, labeltop=False, labelright=False)

        for b, v in zip(bars, model_acc):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2.0, min(v + 0.02, 0.99), f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig_path = out_dir / "fig4_other5_disagree_acc_by_fold.png"
    fig.savefig(fig_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)

    # Print and save 5-fold mean ± std for each model on its own disagree-set.
    print("\n[Disagree-set ACC (other five models disagree): 5-fold mean ± std]")
    rows = []
    for k, name in specs:
        vals = np.asarray(disagree_acc_by_model[k], dtype=np.float64)
        mean, std = _mean_std([float(x) for x in vals if np.isfinite(x)])
        # also report average disagree-set size for context
        n_mean, n_std = _mean_std([float(n) for n in disagree_n_by_model[k]])
        print(f"{name}: {mean:.3f} ± {std:.3f} (n={n_mean:.1f} ± {n_std:.1f})")
        rows.append({"model": k, "name": name, "acc_mean": mean, "acc_std": std, "n_mean": n_mean, "n_std": n_std})
    import pandas as _pd

    _pd.DataFrame(rows).to_csv(out_dir / "fig4_other5_disagree_acc_summary.csv", index=False)


def plot_combined_4rows(out_dir: Path) -> None:
    fig_paths = [
        out_dir / "fig1_confusion_all5fold_concat.png",
        out_dir / "fig3_lda_oof_5fold.png",
        out_dir / "fig2_roc_5fold_by_model.png",
        out_dir / "fig4_other5_disagree_acc_by_fold.png",
    ]
    titles = ["Confusion", "LDA", "ROC", "Disagreement"]

    fig, axes = plt.subplots(4, 1, figsize=(24, 28), constrained_layout=True)
    for ax, p, t in zip(axes, fig_paths, titles):
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(t, loc="left", fontsize=14, pad=6)
        ax.axis("off")

    fig.savefig(out_dir / "fig5_combined_4rows.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_common_style()
    data = _load_oof_data(args)
    _write_model_metrics_summary(data, out_dir)
    lda_scores = _prepare_lda_scores(args, data)

    plot_confusion(data, out_dir, args.model_layout)
    plot_roc(data, out_dir)
    plot_lda(data, lda_scores, out_dir, args.model_layout)
    plot_other4_disagree(data, out_dir)
    plot_combined_4rows(out_dir)

    summarize_kimmig_best_epoch_logs(args.kimmig_log_glob)

    print(f"[OK] wrote 5 figures to {out_dir}")


def summarize_kimmig_best_epoch_logs(log_glob: str) -> None:
    paths = sorted(glob(log_glob))
    if not paths:
        print(f"\n[WARN] no Kimmig logs matched: {log_glob}")
        return

    # fold -> metrics
    metrics_by_fold: dict[int, dict[str, float]] = {}
    for p in paths:
        m = re.search(r"_(\d+)\.out$", str(p))
        if not m:
            continue
        fold = int(m.group(1))

        with open(p, "r", encoding="utf-8", errors="replace") as fp:
            lines = fp.read().splitlines()

        done_json = None
        for line in reversed(lines):
            if "[KIMMIG-BCDB] done" in line and "{" in line:
                done_json = line[line.find("{") :].strip()
                break
        if done_json is None:
            print(f"[WARN] missing done JSON in {p}")
            continue

        try:
            summary = json.loads(done_json)
        except json.JSONDecodeError:
            print(f"[WARN] failed to parse done JSON in {p}")
            continue

        best_epoch = int(summary.get("best_epoch", -1))
        if best_epoch <= 0:
            print(f"[WARN] invalid best_epoch in {p}: {best_epoch}")
            continue

        target = f"fold={fold} epoch={best_epoch:03d}"
        best_line = next((ln for ln in lines if target in ln), None)
        if best_line is None:
            print(f"[WARN] cannot find best-epoch line in {p}: {target}")
            continue

        out: dict[str, float] = {}
        for key in ["ACC", "BACC", "F1", "MCC", "AUC"]:
            mm = re.search(rf"{key}=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", best_line)
            if mm is None:
                out[key] = float("nan")
            else:
                out[key] = float(mm.group(1))

        metrics_by_fold[fold] = out

    want_folds = list(range(5))
    if any(f not in metrics_by_fold for f in want_folds):
        missing = [f for f in want_folds if f not in metrics_by_fold]
        print(f"\n[WARN] missing folds in parsed logs: {missing}")

    rows = []
    for f in want_folds:
        if f in metrics_by_fold:
            r = {"fold": f, **metrics_by_fold[f]}
            rows.append(r)

    if not rows:
        print("\n[WARN] no Kimmig metrics parsed")
        return

    df = pd.DataFrame(rows).sort_values("fold")
    print("\n[SA-GCN best-epoch metrics by fold]")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n[SA-GCN best-epoch metrics: mean ± std over folds]")
    for key in ["ACC", "BACC", "F1", "MCC", "AUC"]:
        vals = df[key].astype(float).values
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean, std = np.nan, np.nan
        elif vals.size == 1:
            mean, std = float(vals[0]), 0.0
        else:
            mean, std = float(vals.mean()), float(vals.std(ddof=1))
        print(f"{key}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
