#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dataloader.dataloader_polymer import build_dataloader
from models.multi_mol_model import MultiMolModel


def parse_args():
    parser = argparse.ArgumentParser(description="BCDB unsupervised representation evaluator")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--pkl_path", required=True, help="Path to BCDB processed pkl")
    parser.add_argument("--output_json", required=True, help="Output json path")
    parser.add_argument("--eval_split", choices=["train", "val", "full"], default="val")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chem_weighting",
        choices=["presence", "dop", "balanced"],
        default="presence",
        help="How to aggregate per-chain chemistry. presence avoids DoP leakage into chemistry.",
    )
    parser.add_argument("--max_pairs", type=int, default=200000)
    parser.add_argument("--topo_weight_vf", type=float, default=1.0)
    parser.add_argument("--topo_weight_logdop", type=float, default=1.0)
    parser.add_argument("--topo_weight_junction", type=float, default=1.0)
    parser.add_argument(
        "--mask_temperature_feature",
        action="store_true",
        help="Mask glob_feat0/chain_glob_feat[0] during embedding extraction.",
    )
    parser.add_argument(
        "--main_task_override",
        choices=["finetune", "chain", "stage2_only"],
        default=None,
        help="Override main_task stored in checkpoint args.",
    )
    return parser.parse_args()


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


def _load_model_and_loader(cli_args):
    ckpt = torch.load(cli_args.checkpoint, map_location="cpu", weights_only=False)
    model_args = _ensure_namespace(ckpt.get("args", {}))

    if not hasattr(model_args, "main_task"):
        raise ValueError("Checkpoint args do not include main_task; cannot infer branch.")

    if cli_args.main_task_override is not None:
        model_args.main_task = cli_args.main_task_override

    if model_args.main_task == "pretrain":
        raise ValueError("Checkpoint main_task=pretrain is not supported for this evaluator.")

    # Dataloader runtime fields
    model_args.distributed = False
    model_args.world_size = 1
    model_args.rank = 0
    model_args.local_rank = 0
    model_args.pin_memory = True
    model_args.num_workers = cli_args.num_workers
    model_args.batch_size = cli_args.batch_size
    model_args.fold = cli_args.fold
    model_args.seed = cli_args.seed

    model = MultiMolModel(model_args)
    state_dict = ckpt.get("model_state", ckpt.get("model", ckpt))
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader, _ = build_dataloader(cli_args.pkl_path, model_args, mode=cli_args.eval_split)
    return model, loader, device, model_args


def _chem_signature(smiles_ids, seg_valid, seg_dop, weighting):
    valid_mask = (seg_valid > 0.5) & (seg_dop > 0)
    ids = smiles_ids[valid_mask].astype(np.int64)
    dops = seg_dop[valid_mask].astype(np.float64)

    if ids.size == 0:
        return {0: 1.0}

    if weighting == "presence":
        unique = np.unique(ids)
        w = 1.0 / float(len(unique))
        return {int(k): w for k in unique.tolist()}

    dop_map = {}
    dop_sum = float(max(dops.sum(), 1e-8))
    for sid, dop in zip(ids.tolist(), dops.tolist()):
        dop_map[int(sid)] = dop_map.get(int(sid), 0.0) + float(dop) / dop_sum

    if weighting == "dop":
        return dop_map

    # balanced = 0.5 * presence + 0.5 * dop share
    unique = np.unique(ids)
    p_map = {int(k): 1.0 / float(len(unique)) for k in unique.tolist()}
    keys = set(dop_map.keys()) | set(p_map.keys())
    out = {}
    for k in keys:
        out[k] = 0.5 * dop_map.get(k, 0.0) + 0.5 * p_map.get(k, 0.0)
    return out


def _sparse_cosine_distance(a, b):
    keys = set(a.keys()) | set(b.keys())
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k in keys:
        va = a.get(k, 0.0)
        vb = b.get(k, 0.0)
        dot += va * vb
        na += va * va
        nb += vb * vb
    denom = math.sqrt(max(na, 1e-12) * max(nb, 1e-12))
    return 1.0 - dot / denom


def _junction_sym(seg_dop, seg_block_id, seg_valid):
    valid = (seg_valid > 0.5) & (seg_dop > 0)
    dop = seg_dop[valid].astype(np.float64)
    blk = seg_block_id[valid].astype(np.int64)
    if dop.size == 0:
        return 0.5
    b0 = dop[blk == 0].sum()
    b1 = dop[blk == 1].sum()
    denom = max(b0 + b1, 1e-8)
    j = float(b0 / denom)
    return min(j, 1.0 - j)


def _vf_sym(block_feat, block_feat_mask):
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


def _rankdata(x):
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


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(max(np.sum(x * x), 1e-12) * max(np.sum(y * y), 1e-12))
    return float(np.sum(x * y) / denom)


def _spearman(x, y):
    return _pearson(_rankdata(np.asarray(x)), _rankdata(np.asarray(y)))


def _residualize(y, controls):
    y = np.asarray(y, dtype=np.float64)
    C = np.asarray(controls, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    X = np.concatenate([np.ones((len(y), 1), dtype=np.float64), C], axis=1)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _partial_corr(y, x, controls):
    ry = _residualize(y, controls)
    rx = _residualize(x, controls)
    return _pearson(ry, rx)


def _sample_pairs(n, max_pairs, seed):
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        return np.triu_indices(n, k=1)

    rng = np.random.default_rng(seed)
    need = int(max_pairs)
    i_all = []
    j_all = []
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


def evaluate(cli_args):
    set_seed(cli_args.seed)
    model, loader, device, model_args = _load_model_and_loader(cli_args)

    rep_holder = {}

    def rep_hook(module, inp, out):
        rep_holder["rep"] = inp[0].detach().cpu().numpy()

    handle = model.reg_head.register_forward_hook(rep_hook)

    reps = []
    chem_signatures = []
    topo_features = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            maybe_mask_temperature_feature(batch, cli_args.mask_temperature_feature)

            rep_holder.clear()
            _ = model(batch)
            if "rep" not in rep_holder:
                raise RuntimeError("Failed to capture reg_head input representation.")
            reps.append(rep_holder["rep"])

            seg_smiles_id = batch["seg_smiles_id"].detach().cpu().numpy()
            seg_valid = batch["seg_valid_mask"].detach().cpu().numpy()
            seg_dop = batch["seg_dop"].detach().cpu().numpy()
            seg_block = batch["seg_block_id"].detach().cpu().numpy()
            block_feat = batch["block_feat"].detach().cpu().numpy()
            block_feat_mask = batch["block_feat_mask"].detach().cpu().numpy()

            for i in range(seg_smiles_id.shape[0]):
                chem_signatures.append(
                    _chem_signature(seg_smiles_id[i], seg_valid[i], seg_dop[i], cli_args.chem_weighting)
                )

                vf_sym = _vf_sym(block_feat[i], block_feat_mask[i])
                log_dop = math.log(max(float(seg_dop[i].sum()), 1.0))
                j_sym = _junction_sym(seg_dop[i], seg_block[i], seg_valid[i])
                topo_features.append([vf_sym, log_dop, j_sym])

    handle.remove()

    reps = np.concatenate(reps, axis=0)
    topo_features = np.asarray(topo_features, dtype=np.float64)

    z = reps.astype(np.float64)
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)

    n = z.shape[0]
    pi, pj = _sample_pairs(n, cli_args.max_pairs, cli_args.seed)

    z_dist = 1.0 - np.sum(z[pi] * z[pj], axis=1)

    # Chemistry distance
    chem_dist = np.array(
        [_sparse_cosine_distance(chem_signatures[i], chem_signatures[j]) for i, j in zip(pi.tolist(), pj.tolist())],
        dtype=np.float64,
    )

    # Topology distance with robust scaling
    scales = np.median(np.abs(topo_features - np.median(topo_features, axis=0, keepdims=True)), axis=0)
    scales = np.clip(scales, 1e-6, None)
    topo_delta = np.abs(topo_features[pi] - topo_features[pj]) / scales[None, :]
    topo_dist = (
        cli_args.topo_weight_vf * topo_delta[:, 0]
        + cli_args.topo_weight_logdop * topo_delta[:, 1]
        + cli_args.topo_weight_junction * topo_delta[:, 2]
    )

    # Collapse metrics
    zc = z - z.mean(axis=0, keepdims=True)
    cov = (zc.T @ zc) / max(zc.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    participation_ratio = float((eigvals.sum() ** 2) / np.sum(eigvals ** 2))
    isotropy_ratio = float(eigvals.min() / eigvals.max())

    # Correlation metrics
    rho_z_chem = _spearman(z_dist, chem_dist)
    rho_z_topo = _spearman(z_dist, topo_dist)
    rho_z_chem_given_topo = _partial_corr(z_dist, chem_dist, topo_dist)
    rho_z_topo_given_chem = _partial_corr(z_dist, topo_dist, chem_dist)

    out = {
        "config": {
            "checkpoint": cli_args.checkpoint,
            "pkl_path": cli_args.pkl_path,
            "eval_split": cli_args.eval_split,
            "fold": cli_args.fold,
            "batch_size": cli_args.batch_size,
            "chem_weighting": cli_args.chem_weighting,
            "mask_temperature_feature": bool(cli_args.mask_temperature_feature),
            "topo_weights": {
                "vf": cli_args.topo_weight_vf,
                "logdop": cli_args.topo_weight_logdop,
                "junction": cli_args.topo_weight_junction,
            },
            "main_task": getattr(model_args, "main_task", "unknown"),
        },
        "dataset": {
            "num_samples": int(n),
            "num_pairs_used": int(len(z_dist)),
        },
        "collapse": {
            "participation_ratio": participation_ratio,
            "isotropy_ratio": isotropy_ratio,
            "mean_pairwise_cosine": float(np.mean(1.0 - z_dist)),
        },
        "alignment": {
            "spearman_latent_vs_chem": float(rho_z_chem),
            "spearman_latent_vs_topology": float(rho_z_topo),
            "partial_latent_chem_given_topology": float(rho_z_chem_given_topo),
            "partial_latent_topology_given_chem": float(rho_z_topo_given_chem),
        },
    }

    out_path = Path(cli_args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
