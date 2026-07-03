#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Post-hoc quantitative analysis + visualization for BCDB unsupervised checkpoints.

Goal: find differences among finetune(stage1only), stage2_only, chain(FRPN) representations.
- Uses *all* samples: dataloader mode = full
- Extracts representation at reg_head input (same as evaluator)
- Reports:
  - collapse stats (participation ratio, isotropy proxy, mean pairwise cosine)
  - distance alignment: Spearman(latent_dist, chem_dist/topo_dist) + partial correlations
  - KNN retrieval curves: neighbor chem/topo distances vs K
  - PCA scatter colored by chemistry/topology proxies

Outputs are written under output_dir.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataloader.dataloader_polymer import build_dataloader
from main import apply_pkl_metadata_to_args
from models.multi_mol_model import MultiMolModel


def parse_args():
    p = argparse.ArgumentParser(description="BCDB unsup post-hoc analysis")
    p.add_argument("--pkl_path", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_temperature_feature", action="store_true")

    p.add_argument("--chem_weighting", choices=["presence", "dop", "balanced"], default="presence")
    p.add_argument("--topo_weight_vf", type=float, default=1.0)
    p.add_argument("--topo_weight_logdop", type=float, default=1.0)
    p.add_argument("--topo_weight_junction", type=float, default=1.0)

    p.add_argument("--max_pairs", type=int, default=2_000_000)
    p.add_argument("--knn_ks", default="1,5,10,20,50")
    p.add_argument("--pairwise_cos_samples", type=int, default=200_000)

    p.add_argument("--ckpt_finetune", required=True)
    p.add_argument("--ckpt_stage2_only", required=True)
    p.add_argument("--ckpt_chain", required=True)

    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_mask_temperature_feature(batch, enabled: bool):
    if not enabled:
        return

    if "glob_feat" in batch and "glob_mask" in batch:
        glob_feat = batch["glob_feat"]
        glob_mask = batch["glob_mask"]
        if torch.is_tensor(glob_feat) and torch.is_tensor(glob_mask) and glob_feat.dim() == 2 and glob_feat.size(1) > 0:
            glob_feat[:, 0] = 0.0
            glob_mask[:, 0] = 0.0
            if "glob_valid_mask" in batch and torch.is_tensor(batch["glob_valid_mask"]):
                batch["glob_valid_mask"] = (glob_mask.sum(dim=1, keepdim=True) > 0).float()

    if "chain_glob_feat" in batch and "chain_glob_mask" in batch:
        c_feat = batch["chain_glob_feat"]
        c_mask = batch["chain_glob_mask"]
        if torch.is_tensor(c_feat) and torch.is_tensor(c_mask) and c_feat.dim() == 2 and c_feat.size(1) > 0:
            c_feat[:, 0] = 0.0
            c_mask[:, 0] = 0.0


def _ensure_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**obj)
    return obj


def _vf_sym(block_feat: np.ndarray, block_feat_mask: np.ndarray) -> float:
    has0 = block_feat_mask[0, 0] > 0.5
    has1 = block_feat_mask[1, 0] > 0.5
    if has0 and has1:
        f1 = float(block_feat[0, 0])
        f2 = float(block_feat[1, 0])
        return min(f1, f2)
    if has0:
        f1 = float(block_feat[0, 0])
        return min(f1, 1.0 - f1)
    if has1:
        f2 = float(block_feat[1, 0])
        return min(f2, 1.0 - f2)
    return 0.5


def _junction_sym(seg_dop: np.ndarray, seg_block_id: np.ndarray, seg_valid: np.ndarray) -> float:
    valid = (seg_valid > 0.5) & (seg_dop > 0)
    dop = seg_dop[valid].astype(np.float64)
    blk = seg_block_id[valid].astype(np.int64)
    if dop.size == 0:
        return 0.5
    b0 = dop[blk == 0].sum()
    b1 = dop[blk == 1].sum()
    denom = max(float(b0 + b1), 1e-8)
    j = float(b0 / denom)
    return min(j, 1.0 - j)


def _chem_dense_vec(
    seg_smiles_id: np.ndarray,
    seg_valid: np.ndarray,
    seg_dop: np.ndarray,
    weighting: str,
    num_types: int,
) -> np.ndarray:
    valid = (seg_valid > 0.5) & (seg_dop > 0)
    ids = seg_smiles_id[valid].astype(np.int64)
    dops = seg_dop[valid].astype(np.float64)

    v = np.zeros((num_types,), dtype=np.float64)
    if ids.size == 0:
        v[0] = 1.0
        return v

    if weighting == "presence":
        uniq = np.unique(ids)
        w = 1.0 / float(len(uniq))
        v[uniq] = w
        return v

    dop_sum = float(max(dops.sum(), 1e-8))
    for sid, dop in zip(ids.tolist(), dops.tolist()):
        if 0 <= int(sid) < num_types:
            v[int(sid)] += float(dop) / dop_sum

    if weighting == "dop":
        return v

    uniq = np.unique(ids)
    p = np.zeros((num_types,), dtype=np.float64)
    p[uniq] = 1.0 / float(len(uniq))
    return 0.5 * v + 0.5 * p


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)

    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        if end - start > 1:
            avg_rank = 0.5 * (start + end - 1)
            ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(max(np.sum(x * x), 1e-12) * max(np.sum(y * y), 1e-12))
    return float(np.sum(x * y) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(np.asarray(x)), _rankdata(np.asarray(y)))


def _residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    C = np.asarray(controls, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    X = np.concatenate([np.ones((len(y), 1), dtype=np.float64), C], axis=1)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _partial_corr(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    ry = _residualize(y, controls)
    rx = _residualize(x, controls)
    return _pearson(ry, rx)


def _sample_pairs(n: int, max_pairs: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        return np.triu_indices(n, k=1)

    rng = np.random.default_rng(seed)
    need = int(max_pairs)
    i_all: List[np.ndarray] = []
    j_all: List[np.ndarray] = []
    while need > 0:
        m = int(max(need * 2, 4096))
        i = rng.integers(0, n, size=m)
        j = rng.integers(0, n, size=m)
        keep = i < j
        i = i[keep]
        j = j[keep]
        if i.size == 0:
            continue
        take = min(need, i.size)
        i_all.append(i[:take])
        j_all.append(j[:take])
        need -= take

    return np.concatenate(i_all), np.concatenate(j_all)


def _participation_ratio(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    vals = np.linalg.eigvalsh(cov)
    vals = np.clip(vals, 1e-12, None)
    return float((vals.sum() ** 2) / (np.sum(vals**2) + 1e-12))


def _effective_rank_from_eigs(eigs: np.ndarray) -> float:
    eigs = np.asarray(eigs, dtype=np.float64)
    eigs = np.clip(eigs, 1e-15, None)
    p = eigs / eigs.sum()
    h = -np.sum(p * np.log(p))
    return float(np.exp(h))


def spectrum_stats(z: np.ndarray, max_k: int = 50) -> Dict:
    """Compute variance spectrum on centered raw reps (no L2-norm).

    Returns top eigenvalues (cov) and cumulative explained variance.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    n = z.shape[0]
    if n < 2:
        return {"eigs": [], "cum_explained": [], "pc1_explained": 1.0, "effective_rank": 1.0}

    # Use SVD on data matrix; eigenvalues of covariance are (s^2)/(n-1)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xt = torch.from_numpy(z.astype(np.float32)).to(device)
        # torch.linalg.svd is stable; for (n=5k,d=768) it is fine on GPU
        s = torch.linalg.svdvals(xt)
        s = s.detach().cpu().numpy().astype(np.float64)
    except Exception:
        s = np.linalg.svd(z, compute_uv=False).astype(np.float64)

    eigs = (s * s) / max(n - 1, 1)
    total = float(np.sum(eigs) + 1e-15)
    expl = eigs / total

    k = int(min(max_k, expl.shape[0]))
    cum = np.cumsum(expl[:k])

    return {
        "eigs": eigs[:k].tolist(),
        "explained": expl[:k].tolist(),
        "cum_explained": cum.tolist(),
        "pc1_explained": float(expl[0]) if expl.size else 1.0,
        "effective_rank": _effective_rank_from_eigs(eigs),
    }


def cosine_sim_stats(z_norm: np.ndarray, num_pairs: int, seed: int) -> Dict:
    z = np.asarray(z_norm, dtype=np.float64)
    n = z.shape[0]
    if n < 2:
        return {"mean": 1.0, "std": 0.0, "p01": 1.0, "p05": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0}

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    keep = i != j
    i = i[keep]
    j = j[keep]
    if i.size == 0:
        return {"mean": 1.0, "std": 0.0, "p01": 1.0, "p05": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0}

    sims = np.sum(z[i] * z[j], axis=1)
    return {
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "p01": float(np.quantile(sims, 0.01)),
        "p05": float(np.quantile(sims, 0.05)),
        "p50": float(np.quantile(sims, 0.50)),
        "p95": float(np.quantile(sims, 0.95)),
        "p99": float(np.quantile(sims, 0.99)),
    }


def _mean_pairwise_cosine(z: np.ndarray, num_pairs: int, seed: int) -> float:
    n = z.shape[0]
    if n < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    keep = i != j
    i = i[keep]
    j = j[keep]
    if i.size == 0:
        return 1.0
    cos = np.sum(z[i] * z[j], axis=1)
    return float(np.mean(cos))


@torch.no_grad()
def extract_reps_and_meta(
    checkpoint_path: str,
    pkl_path: str,
    fold: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    mask_temperature_feature: bool,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = _ensure_namespace(ckpt.get("args", {}))

    # runtime dataloader fields
    model_args.distributed = False
    model_args.world_size = 1
    model_args.rank = 0
    model_args.local_rank = 0
    model_args.pin_memory = True
    model_args.num_workers = num_workers
    model_args.batch_size = batch_size
    model_args.fold = fold
    model_args.seed = seed

    # Needed for apply_pkl_metadata_to_args
    model_args.pkl_path = pkl_path

    # ensure vocab / max_segments consistent with pkl
    apply_pkl_metadata_to_args(model_args)

    model = MultiMolModel(model_args)
    state_dict = ckpt.get("model_state", ckpt.get("model", ckpt))
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader, _ = build_dataloader(pkl_path, model_args, mode="full")

    rep_holder: Dict[str, np.ndarray] = {}

    def rep_hook(module, inp, out):
        rep_holder["rep"] = inp[0].detach().cpu().numpy()

    handle = model.reg_head.register_forward_hook(rep_hook)

    reps: List[np.ndarray] = []
    seg_smiles_all: List[np.ndarray] = []
    seg_valid_all: List[np.ndarray] = []
    seg_dop_all: List[np.ndarray] = []
    seg_block_all: List[np.ndarray] = []
    block_feat_all: List[np.ndarray] = []
    block_feat_mask_all: List[np.ndarray] = []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        maybe_mask_temperature_feature(batch, mask_temperature_feature)

        rep_holder.clear()
        _ = model(batch)
        if "rep" not in rep_holder:
            raise RuntimeError("Failed to capture reg_head input representation.")

        reps.append(rep_holder["rep"])
        seg_smiles_all.append(batch["seg_smiles_id"].detach().cpu().numpy())
        seg_valid_all.append(batch["seg_valid_mask"].detach().cpu().numpy())
        seg_dop_all.append(batch["seg_dop"].detach().cpu().numpy())
        seg_block_all.append(batch["seg_block_id"].detach().cpu().numpy())
        block_feat_all.append(batch["block_feat"].detach().cpu().numpy())
        block_feat_mask_all.append(batch["block_feat_mask"].detach().cpu().numpy())

    handle.remove()

    rep = np.concatenate(reps, axis=0).astype(np.float32)
    seg_smiles_id = np.concatenate(seg_smiles_all, axis=0)
    seg_valid = np.concatenate(seg_valid_all, axis=0)
    seg_dop = np.concatenate(seg_dop_all, axis=0)
    seg_block = np.concatenate(seg_block_all, axis=0)
    block_feat = np.concatenate(block_feat_all, axis=0)
    block_feat_mask = np.concatenate(block_feat_mask_all, axis=0)

    return rep, seg_smiles_id, seg_valid, seg_dop, seg_block, block_feat, block_feat_mask, model_args


def compute_chem_topo_arrays(
    seg_smiles_id: np.ndarray,
    seg_valid: np.ndarray,
    seg_dop: np.ndarray,
    seg_block: np.ndarray,
    block_feat: np.ndarray,
    block_feat_mask: np.ndarray,
    weighting: str,
    num_types: int,
):
    n = seg_smiles_id.shape[0]
    chem = np.zeros((n, num_types), dtype=np.float32)
    topo = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        v = _chem_dense_vec(seg_smiles_id[i], seg_valid[i], seg_dop[i], weighting, num_types)
        chem[i] = v.astype(np.float32)

        topo[i, 0] = _vf_sym(block_feat[i], block_feat_mask[i])
        topo[i, 1] = float(math.log(max(float(seg_dop[i].sum()), 1.0)))
        topo[i, 2] = _junction_sym(seg_dop[i], seg_block[i], seg_valid[i])

    # cosine distance uses L2-normalized chem vectors
    chem_norm = chem / (np.linalg.norm(chem, axis=1, keepdims=True) + 1e-12)

    topo_med = np.median(topo, axis=0, keepdims=True)
    topo_scales = np.median(np.abs(topo - topo_med), axis=0)
    topo_scales = np.clip(topo_scales, 1e-6, None).astype(np.float32)

    return chem_norm.astype(np.float32), topo.astype(np.float32), topo_scales


def compute_distance_arrays_for_pairs(
    z_norm: np.ndarray,
    chem_norm: np.ndarray,
    topo: np.ndarray,
    topo_scales: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    topo_weights: Tuple[float, float, float],
):
    latent_dist = 1.0 - np.sum(z_norm[i_idx] * z_norm[j_idx], axis=1)
    chem_dist = 1.0 - np.sum(chem_norm[i_idx] * chem_norm[j_idx], axis=1)

    dvf = np.abs(topo[i_idx, 0] - topo[j_idx, 0]) / topo_scales[0]
    dlogdop = np.abs(topo[i_idx, 1] - topo[j_idx, 1]) / topo_scales[1]
    dj = np.abs(topo[i_idx, 2] - topo[j_idx, 2]) / topo_scales[2]
    topo_dist = topo_weights[0] * dvf + topo_weights[1] * dlogdop + topo_weights[2] * dj

    return latent_dist.astype(np.float64), chem_dist.astype(np.float64), topo_dist.astype(np.float64)


@torch.no_grad()
def knn_retrieval_curves(
    z_norm: np.ndarray,
    chem_norm: np.ndarray,
    topo: np.ndarray,
    topo_scales: np.ndarray,
    topo_weights: Tuple[float, float, float],
    ks: List[int],
    device: str = "cuda",
    chunk: int = 1024,
):
    z = torch.from_numpy(z_norm).to(device=device, dtype=torch.float16)
    chem = torch.from_numpy(chem_norm).to(device=device, dtype=torch.float16)
    topo_t = torch.from_numpy(topo).to(device=device, dtype=torch.float32)
    scales = torch.from_numpy(topo_scales).to(device=device, dtype=torch.float32)

    n = z.size(0)
    max_k = int(max(ks))

    sum_chem = {k: 0.0 for k in ks}
    sum_topo = {k: 0.0 for k in ks}
    sum_dvf = {k: 0.0 for k in ks}
    sum_dlogdop = {k: 0.0 for k in ks}
    sum_dj = {k: 0.0 for k in ks}

    wvf, wlogdop, wj = [float(x) for x in topo_weights]

    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        zq = z[start:end]  # [q, d]
        sims = (zq @ z.t()).to(torch.float32)  # [q, n]

        # topk includes self; we will drop it
        topv, topi = torch.topk(sims, k=min(max_k + 1, n), dim=1, largest=True, sorted=True)
        nbrs = topi[:, 1 : max_k + 1]  # [q, max_k]

        chem_q = chem[start:end].to(torch.float32)  # [q, c]
        chem_n = chem[nbrs].to(torch.float32)  # [q, max_k, c]
        chem_dot = (chem_n * chem_q.unsqueeze(1)).sum(dim=-1)
        chem_dist = 1.0 - chem_dot  # [q, max_k]

        topo_q = topo_t[start:end].unsqueeze(1)  # [q, 1, 3]
        topo_n = topo_t[nbrs]  # [q, max_k, 3]
        d = (topo_q - topo_n).abs()
        d[..., 0] = d[..., 0] / scales[0]
        d[..., 1] = d[..., 1] / scales[1]
        d[..., 2] = d[..., 2] / scales[2]
        topo_dist = wvf * d[..., 0] + wlogdop * d[..., 1] + wj * d[..., 2]  # [q, max_k]

        for k in ks:
            kk = int(k)
            sum_chem[k] += float(chem_dist[:, :kk].mean().item()) * (end - start)
            sum_topo[k] += float(topo_dist[:, :kk].mean().item()) * (end - start)
            sum_dvf[k] += float(d[..., 0][:, :kk].mean().item()) * (end - start)
            sum_dlogdop[k] += float(d[..., 1][:, :kk].mean().item()) * (end - start)
            sum_dj[k] += float(d[..., 2][:, :kk].mean().item()) * (end - start)

    out = {
        "ks": ks,
        "chem_mean_dist": [sum_chem[k] / float(n) for k in ks],
        "topo_mean_dist": [sum_topo[k] / float(n) for k in ks],
        "topo_components": {
            "dvf": [sum_dvf[k] / float(n) for k in ks],
            "dlogdop": [sum_dlogdop[k] / float(n) for k in ks],
            "djunction": [sum_dj[k] / float(n) for k in ks],
        },
    }
    return out


def baseline_pair_means(
    z_norm: np.ndarray,
    chem_norm: np.ndarray,
    topo: np.ndarray,
    topo_scales: np.ndarray,
    topo_weights: Tuple[float, float, float],
    num_pairs: int,
    seed: int,
) -> Dict:
    n = int(z_norm.shape[0])
    i, j = _sample_pairs(n, num_pairs, seed)
    latent_dist, chem_dist, topo_dist = compute_distance_arrays_for_pairs(
        z_norm, chem_norm, topo, topo_scales, i, j, topo_weights
    )

    dvf = np.abs(topo[i, 0] - topo[j, 0]) / topo_scales[0]
    dlogdop = np.abs(topo[i, 1] - topo[j, 1]) / topo_scales[1]
    dj = np.abs(topo[i, 2] - topo[j, 2]) / topo_scales[2]

    return {
        "latent_mean": float(np.mean(latent_dist)),
        "chem_mean": float(np.mean(chem_dist)),
        "topo_mean": float(np.mean(topo_dist)),
        "topo_components": {
            "dvf": float(np.mean(dvf)),
            "dlogdop": float(np.mean(dlogdop)),
            "djunction": float(np.mean(dj)),
        },
    }


def pca_2d(x: np.ndarray, seed: int) -> np.ndarray:
    # Prefer torch.pca_lowrank for speed; fallback to numpy SVD.
    try:
        torch.manual_seed(int(seed))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        xt = xt - xt.mean(dim=0, keepdim=True)
        # q=2 is sufficient for 2D scatter
        _, _, v = torch.pca_lowrank(xt, q=2, center=False)
        xy = (xt @ v[:, :2]).detach().cpu().numpy()
        return xy.astype(np.float32)
    except Exception:
        x64 = np.asarray(x, dtype=np.float64)
        x64 = x64 - x64.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x64, full_matrices=False)
        comp = vt[:2].T  # [d, 2]
        return (x64 @ comp).astype(np.float32)


def plot_all(output_dir: Path, results: Dict):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] plotting skipped (matplotlib unavailable): {e}")
        return

    # --- KNN curves ---
    ks = results["knn"]["finetune"]["ks"]
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for name, color in [("finetune", "C0"), ("stage2_only", "C1"), ("chain", "C2")]:
        ax1.plot(ks, results["knn"][name]["chem_mean_dist"], label=name, color=color)
        ax2.plot(ks, results["knn"][name]["topo_mean_dist"], label=name, color=color)

    ax1.set_title("KNN neighbor chemistry distance")
    ax2.set_title("KNN neighbor topology distance")
    for ax in (ax1, ax2):
        ax.set_xlabel("K")
        ax.set_ylabel("mean distance (lower is better)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / "knn_curves.png", dpi=220)
    plt.close(fig)

    # --- Baseline-normalized KNN curves (divide by random-pair mean) ---
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for name, color in [("finetune", "C0"), ("stage2_only", "C1"), ("chain", "C2")]:
        b_chem = float(results["baseline"][name]["chem_mean"])
        b_topo = float(results["baseline"][name]["topo_mean"])
        chem = np.asarray(results["knn"][name]["chem_mean_dist"], dtype=np.float64) / max(b_chem, 1e-12)
        topo = np.asarray(results["knn"][name]["topo_mean_dist"], dtype=np.float64) / max(b_topo, 1e-12)
        ax1.plot(ks, chem, label=name, color=color)
        ax2.plot(ks, topo, label=name, color=color)

    ax1.set_title("KNN chem dist / random mean")
    ax2.set_title("KNN topo dist / random mean")
    for ax in (ax1, ax2):
        ax.set_xlabel("K")
        ax.set_ylabel("ratio (<1 means better than random)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / "knn_curves_norm.png", dpi=220)
    plt.close(fig)

    # --- PC1 correlations bar plot ---
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    feats = ["chem_entropy", "logdop", "vf_sym", "junction_sym"]
    x = np.arange(len(feats))
    width = 0.26
    for idx, (name, color) in enumerate([("finetune", "C0"), ("stage2_only", "C1"), ("chain", "C2")]):
        c = results["pc1_corr"][name]
        vals = [
            float(c["pearson_pc1_chem_entropy"]),
            float(c["pearson_pc1_logdop"]),
            float(c["pearson_pc1_vf_sym"]),
            float(c["pearson_pc1_junction_sym"]),
        ]
        ax.bar(x + (idx - 1) * width, vals, width=width, label=name, color=color, alpha=0.9)

    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(feats)
    ax.set_title("Pearson corr(PC1, factor)")
    ax.set_ylabel("correlation")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pc1_corr.png", dpi=220)
    plt.close(fig)

    # --- Spectrum plot ---
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for name, color in [("finetune", "C0"), ("stage2_only", "C1"), ("chain", "C2")]:
        spec = results["spectrum"][name]
        expl = np.asarray(spec.get("explained", []), dtype=np.float64)
        cum = np.asarray(spec.get("cum_explained", []), dtype=np.float64)
        if expl.size:
            ax1.plot(np.arange(1, expl.size + 1), expl, label=name, color=color)
            ax2.plot(np.arange(1, cum.size + 1), cum, label=name, color=color)

    ax1.set_title("Explained variance ratio (top PCs)")
    ax2.set_title("Cumulative explained variance")
    for ax in (ax1, ax2):
        ax.set_xlabel("PC index")
        ax.set_ylabel("ratio")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / "spectrum.png", dpi=220)
    plt.close(fig)

    # --- PCA scatter (chem entropy / logDoP) ---
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 2, wspace=0.15, hspace=0.18)

    for r, name in enumerate(["finetune", "stage2_only", "chain"]):
        xy = np.asarray(results["pca2"][name], dtype=np.float32)
        chem_entropy = np.asarray(results["meta"]["chem_entropy"], dtype=np.float32)
        logdop = np.asarray(results["meta"]["topo"]["logdop"], dtype=np.float32)

        ax = fig.add_subplot(gs[r, 0])
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=chem_entropy, s=4, alpha=0.7, cmap="viridis")
        ax.set_title(f"{name}: PCA colored by chem entropy")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.01)
        cbar.ax.tick_params(labelsize=8)

        ax = fig.add_subplot(gs[r, 1])
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=logdop, s=4, alpha=0.7, cmap="magma")
        ax.set_title(f"{name}: PCA colored by logDoP")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.01)
        cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "pca_scatter.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpts = {
        "finetune": args.ckpt_finetune,
        "stage2_only": args.ckpt_stage2_only,
        "chain": args.ckpt_chain,
    }

    # Extract meta from one model pass; for full split it is identical across ckpts.
    # Use finetune checkpoint for meta extraction.
    print("[STEP] extract finetune reps + meta (full)")
    rep0, seg_smiles_id, seg_valid, seg_dop, seg_block, block_feat, block_feat_mask, model_args0 = extract_reps_and_meta(
        ckpts["finetune"],
        args.pkl_path,
        args.fold,
        args.batch_size,
        args.num_workers,
        args.seed,
        args.mask_temperature_feature,
    )
    print(f"[OK] finetune reps: {rep0.shape}")

    num_types = int(getattr(model_args0, "num_seg_smiles_types", 20000))
    num_types = int(min(num_types, int(np.max(seg_smiles_id) + 1) if seg_smiles_id.size > 0 else num_types))

    chem_norm, topo, topo_scales = compute_chem_topo_arrays(
        seg_smiles_id,
        seg_valid,
        seg_dop,
        seg_block,
        block_feat,
        block_feat_mask,
        args.chem_weighting,
        num_types,
    )

    # proxy labels for visualization
    chem_p = np.clip(chem_norm, 0.0, None)
    chem_p = chem_p / (chem_p.sum(axis=1, keepdims=True) + 1e-12)
    chem_entropy = -np.sum(chem_p * np.log(np.clip(chem_p, 1e-12, None)), axis=1).astype(np.float32)

    meta = {
        "n_samples": int(chem_norm.shape[0]),
        "chem_entropy": chem_entropy.tolist(),
        "topo": {
            "vf_sym": topo[:, 0].astype(np.float32).tolist(),
            "logdop": topo[:, 1].astype(np.float32).tolist(),
            "junction_sym": topo[:, 2].astype(np.float32).tolist(),
            "scales": topo_scales.astype(np.float32).tolist(),
        },
    }

    topo_weights = (args.topo_weight_vf, args.topo_weight_logdop, args.topo_weight_junction)

    # Shared sampled pairs for all models
    i_idx, j_idx = _sample_pairs(meta["n_samples"], args.max_pairs, args.seed)

    results = {
        "config": {
            "pkl_path": args.pkl_path,
            "fold": args.fold,
            "chem_weighting": args.chem_weighting,
            "topo_weights": {
                "vf": args.topo_weight_vf,
                "logdop": args.topo_weight_logdop,
                "junction": args.topo_weight_junction,
            },
            "max_pairs": int(args.max_pairs),
            "knn_ks": args.knn_ks,
            "mask_temperature_feature": bool(args.mask_temperature_feature),
            "checkpoints": ckpts,
        },
        "meta": meta,
        "alignment": {},
        "collapse": {},
        "knn": {},
        "pca2": {},
        "pc1_corr": {},
        "spectrum": {},
        "baseline": {},
        "cosine_sim": {},
    }

    ks = [int(x) for x in args.knn_ks.split(",") if x.strip()]

    # For finetune rep0 already extracted; reuse
    reps_by_model = {"finetune": rep0}

    for name in ["stage2_only", "chain"]:
        print(f"[STEP] extract {name} reps (full)")
        rep, *_ = extract_reps_and_meta(
            ckpts[name],
            args.pkl_path,
            args.fold,
            args.batch_size,
            args.num_workers,
            args.seed,
            args.mask_temperature_feature,
        )
        reps_by_model[name] = rep
        print(f"[OK] {name} reps: {rep.shape}")

    for name, rep in reps_by_model.items():
        print(f"[STEP] metrics for {name}")
        z = rep.astype(np.float32)
        z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

        results["spectrum"][name] = spectrum_stats(z, max_k=50)
        results["cosine_sim"][name] = cosine_sim_stats(z_norm, num_pairs=min(200000, int(args.max_pairs)), seed=args.seed)

        latent_dist, chem_dist, topo_dist = compute_distance_arrays_for_pairs(
            z_norm,
            chem_norm,
            topo,
            topo_scales,
            i_idx,
            j_idx,
            topo_weights,
        )

        align = {
            "spearman_latent_vs_chem": _spearman(latent_dist, chem_dist),
            "spearman_latent_vs_topology": _spearman(latent_dist, topo_dist),
            "partial_latent_chem_given_topology": _partial_corr(latent_dist, chem_dist, topo_dist),
            "partial_latent_topology_given_chem": _partial_corr(latent_dist, topo_dist, chem_dist),
        }
        results["alignment"][name] = align
        print(f"[OK] {name} align: spearman(chem)={align['spearman_latent_vs_chem']:.4f} spearman(topo)={align['spearman_latent_vs_topology']:.4f}")

        collapse = {
            "participation_ratio": _participation_ratio(z_norm),
            "mean_pairwise_cosine": _mean_pairwise_cosine(z_norm, args.pairwise_cos_samples, args.seed),
        }
        results["collapse"][name] = collapse
        print(f"[OK] {name} collapse: pr={collapse['participation_ratio']:.3f} mean_cos={collapse['mean_pairwise_cosine']:.3f}")

        results["knn"][name] = knn_retrieval_curves(
            z_norm,
            chem_norm,
            topo,
            topo_scales,
            topo_weights,
            ks,
        )

        print(f"[OK] {name} knn curves computed")

        xy = pca_2d(z_norm, seed=args.seed)
        results["pca2"][name] = xy.tolist()
        # Correlate PC1 with interpretable meta factors
        pc1 = xy[:, 0].astype(np.float64)
        results["pc1_corr"][name] = {
            "pearson_pc1_chem_entropy": _pearson(pc1, chem_entropy.astype(np.float64)),
            "pearson_pc1_logdop": _pearson(pc1, topo[:, 1].astype(np.float64)),
            "pearson_pc1_vf_sym": _pearson(pc1, topo[:, 0].astype(np.float64)),
            "pearson_pc1_junction_sym": _pearson(pc1, topo[:, 2].astype(np.float64)),
        }
        print(f"[OK] {name} pca2 computed")

        # Save embeddings for later reuse
        np.savez_compressed(
            output_dir / f"emb_{name}.npz",
            rep=z.astype(np.float16),
            rep_norm=z_norm.astype(np.float16),
        )

        # Baseline means for interpreting KNN distances
        results["baseline"][name] = baseline_pair_means(
            z_norm,
            chem_norm,
            topo,
            topo_scales,
            topo_weights,
            num_pairs=min(200000, int(args.max_pairs)),
            seed=args.seed + 17,
        )

    with (output_dir / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)

    plot_all(output_dir, results)

    print(f"[OK] wrote summary -> {output_dir / 'summary.json'}")
    print(f"[OK] wrote figs -> {output_dir / 'knn_curves.png'} and {output_dir / 'pca_scatter.png'}")


if __name__ == "__main__":
    main()
