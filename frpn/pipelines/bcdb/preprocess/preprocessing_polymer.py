# preprocessing/preprocessing_polymer.py
# Combined preprocessing script supporting multiple SMILES segments and full graph features.

from __future__ import annotations

import random
import argparse
import hashlib
import sys
import pickle
import gc
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from multiprocessing import get_context
from multiprocessing.dummy import Pool as ThreadPool

from molecular_features import build_initial_graph, build_graph_features

RDLogger.DisableLog("rdApp.*")
RDLogger.DisableLog("rdWarning.*")
RDLogger.DisableLog("rdError.*")
torch.multiprocessing.set_sharing_strategy("file_system")

#### SEQUENCE CONFIGS ####
MAX_SEGMENTS = 5
MAX_LOCAL_FEATS = 2
MAX_GLOBAL_FEATS = 2

#### MULTI-CONF CONFIGS ####
NUM_CONFS = 4
MAX_TRIES = NUM_CONFS * 2


def _dump_one(
    i: int,
    samples: List[Dict[str, Any]],
    outdir: Path,
    label_cols: List[str],
    max_segments: int,
) -> None:
    header = {
        "glob_feature_count": MAX_GLOBAL_FEATS,
        "local_feature_count": MAX_LOCAL_FEATS,
        "max_segments": max_segments,
        "label_names": label_cols,
        "samples": samples,
    }
    with (outdir / f"shard{i}.pkl").open("wb") as f:
        pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] wrote shard{i}.pkl ({len(samples)} samples)")


def _replace_star_with_dummyC(mol: Chem.Mol) -> List[int]:
    star_idx: List[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(6)
            star_idx.append(atom.GetIdx())
    return star_idx


def _restore_dummy(mol: Chem.Mol, idx_list: List[int]) -> None:
    for idx in idx_list:
        mol.GetAtomWithIdx(idx).SetAtomicNum(0)


def _embed_and_strip_h(smiles: str) -> Tuple[Chem.Mol, np.ndarray] | None:
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        return None

    def _compute_2d_no_h() -> Tuple[Chem.Mol, np.ndarray]:
        mol2d = Chem.AddHs(mol0)
        tmp2d = Chem.Mol(mol2d)

        changed = _replace_star_with_dummyC(tmp2d)
        AllChem.Compute2DCoords(tmp2d)
        _restore_dummy(tmp2d, changed)

        no_h_2d = Chem.RemoveAllHs(tmp2d)
        coords_2d = no_h_2d.GetConformer().GetPositions().astype(np.float32)
        assert len(no_h_2d.GetAtoms()) == coords_2d.shape[0], (
            f"2D coordinates shape is not aligned with {smiles}"
        )
        return no_h_2d, coords_2d

    mol_with_h = Chem.AddHs(mol0)
    conformer_coords: List[np.ndarray] = []

    tries = 0
    base_seed = 42
    no_h_3d: Chem.Mol | None = None

    while len(conformer_coords) < NUM_CONFS and tries < MAX_TRIES:
        attempt_seed = base_seed + tries * 42
        tmp = Chem.Mol(mol_with_h)
        params = AllChem.ETKDGv3()
        params.randomSeed = attempt_seed
        params.maxIterations = 42
        try:
            changed = _replace_star_with_dummyC(tmp)
            AllChem.EmbedMolecule(tmp, params)
            AllChem.MMFFOptimizeMolecule(tmp)
            _restore_dummy(tmp, changed)

            no_h = Chem.RemoveAllHs(tmp)
            pos = no_h.GetConformer().GetPositions().astype(np.float32)
            conformer_coords.append(pos)
            no_h_3d = no_h
        except Exception:
            pass
        tries += 1

    if len(conformer_coords) == 0:
        try:
            no_h_2d, coords_2d = _compute_2d_no_h()
        except Exception:
            return None
        coords_arr = np.repeat(coords_2d[None, ...], NUM_CONFS, axis=0)
        return no_h_2d, coords_arr

    if len(conformer_coords) < NUM_CONFS:
        try:
            no_h_2d, coords_2d = _compute_2d_no_h()
            if coords_2d.shape == conformer_coords[0].shape:
                conformer_coords.append(coords_2d)
                if no_h_3d is None:
                    no_h_3d = no_h_2d
        except Exception:
            pass

    coords_arr = np.stack(conformer_coords, axis=0)
    if coords_arr.shape[0] < NUM_CONFS:
        pad_n = NUM_CONFS - coords_arr.shape[0]
        pad = np.repeat(coords_arr[-1:, ...], pad_n, axis=0)
        coords_arr = np.concatenate([coords_arr, pad], axis=0)

    if no_h_3d is None:
        try:
            no_h_3d, _ = _compute_2d_no_h()
        except Exception:
            return None

    return no_h_3d, coords_arr


def _build_numeric_tensor(
    vals: List[Any], dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    data: List[float] = []
    mask: List[int] = []
    for v in vals:
        if v is None:
            data.append(0.0)
            mask.append(0)
        else:
            data.append(float(v))
            mask.append(1)
    return torch.tensor(data, dtype=dtype), torch.tensor(mask, dtype=dtype)


def _process_row(args: tuple) -> Dict[str, Any]:
    row_idx, row, smiles_cols, label_cols, max_segments, volume_mode, smiles_vocab = args

    smiles_entries: List[Tuple[int, str]] = [
        (i, row[c])
        for i, c in enumerate(smiles_cols)
        if pd.notna(row[c]) and row[c] != ""
    ]
    if not smiles_entries:
        return {"_error": True, "row_idx": row_idx, "failed_smi": None}

    smiles_list: List[str] = [smi for _, smi in smiles_entries]

    seg_smiles_id = torch.zeros(max_segments, dtype=torch.long)
    for seg_idx, smi in smiles_entries:
        seg_smiles_id[seg_idx] = int(smiles_vocab.get(smi, 0))

    segment_ids: List[int] = []
    coords_list: List[np.ndarray] = []
    node_feats_all: List[np.ndarray] = []
    edge_feats_all: List[torch.Tensor] = []
    sp_all: List[torch.Tensor] = []
    degree_all: List[torch.Tensor] = []
    atom_mask_all: List[torch.Tensor] = []
    atom_counts: List[int] = []
    atomic_nums_all: List[torch.Tensor] = []

    for seg_idx, smi in smiles_entries:
        res = _embed_and_strip_h(smi)
        if res is None:
            return {"_error": True, "row_idx": row_idx, "failed_smi": smi}
        mol, coords_arr = res
        node_attr, edge_index, edge_attr, proc_mol, atomic_nums = build_initial_graph(mol)
        atomic_nums_all.append(torch.from_numpy(atomic_nums).long())
        feat = build_graph_features(node_attr, edge_index, edge_attr, proc_mol)

        N = node_attr.shape[0]
        atom_counts.append(N)
        segment_ids.extend([seg_idx] * N)

        coords_list.append(coords_arr)

        edge_feats_all.append(feat["edge_feat"])
        sp_all.append(feat["shortest_path"])
        degree_all.append(feat["degree"])
        atom_mask_all.append(feat["atom_mask"])
        node_feats_all.append(feat["atom_feat"])

    total_N = int(sum(atom_counts))

    atom_token = torch.cat(atomic_nums_all, dim=0)

    coords_per_k: List[np.ndarray] = []
    for k in range(NUM_CONFS):
        parts = [segment_coords[k] for segment_coords in coords_list]
        coords_per_k.append(np.vstack(parts))
    coords_arr_all = np.stack(coords_per_k, axis=0)
    src_pos = torch.from_numpy(coords_arr_all)

    atom_mask = torch.cat(atom_mask_all, dim=0)

    segment_id = torch.tensor(segment_ids, dtype=torch.long)

    B = int(edge_feats_all[0].shape[-1])
    edge_feat = torch.full((total_N, total_N, B), 2, dtype=edge_feats_all[0].dtype)
    shortest_path = torch.full((total_N, total_N), 510, dtype=sp_all[0].dtype)
    degree = torch.zeros((total_N,), dtype=degree_all[0].dtype)

    BASE = 128
    z_idx = torch.cat(atomic_nums_all, dim=0).long()
    zi = z_idx.view(-1, 1).expand(total_N, total_N)
    zj = z_idx.view(1, -1).expand(total_N, total_N)
    pair_type = zi * BASE + zj

    atom_feat = torch.cat(node_feats_all, dim=0)

    idx_start = 0
    for i, N in enumerate(atom_counts):
        i0 = idx_start
        i1 = idx_start + N
        edge_feat[i0:i1, i0:i1, :] = edge_feats_all[i]
        shortest_path[i0:i1, i0:i1] = sp_all[i]
        degree[i0:i1] = degree_all[i]
        idx_start += N

    glob_feat = [
        None if f"glob_feat{i}" not in row or pd.isna(row[f"glob_feat{i}"]) else row[f"glob_feat{i}"]
        for i in range(MAX_GLOBAL_FEATS)
    ]

    glob_feat_tensor, glob_mask_tensor = _build_numeric_tensor(glob_feat)
    glob_valid_mask = (glob_mask_tensor.sum() > 0).float()

    # ---- seg dop & block id ----
    seg_dop = torch.zeros(max_segments, dtype=torch.long)
    seg_block_id = torch.zeros(max_segments, dtype=torch.long)

    for s in range(max_segments):
        dop_col = f"seg{s}_dop_raw"
        blk_col = f"seg{s}_block_id"

        if dop_col in row and not pd.isna(row[dop_col]):
            seg_dop[s] = int(row[dop_col])
        elif s == 0:
            # Homopolymer / simple polymer datasets may provide DP (degree of polymerization)
            # instead of the generic seg{s}_dop_raw columns. Treat DP as seg0_dop_raw.
            if "DP" in row and not pd.isna(row["DP"]):
                seg_dop[s] = int(row["DP"])
            elif "seg0_feat0" in row and not pd.isna(row["seg0_feat0"]):
                # Some datasets store DP directly in seg0_feat0.
                seg_dop[s] = int(row["seg0_feat0"])

        if blk_col in row and not pd.isna(row[blk_col]):
            seg_block_id[s] = int(row[blk_col])

    seg_feat_tensor = torch.zeros(max_segments, MAX_LOCAL_FEATS)
    seg_feat_mask = torch.zeros_like(seg_feat_tensor)
    seg_valid_mask = torch.zeros(max_segments, dtype=torch.float32)

    active_seg_idx = torch.unique(segment_id).long().tolist()
    for s in active_seg_idx:
        if 0 <= s < max_segments:
            seg_valid_mask[s] = 1.0

    # seg feature #1: DoP share (kept in both with/without-volume modes)
    dop_sum = float(seg_dop.float().sum().item())
    if dop_sum > 0:
        for s in active_seg_idx:
            seg_feat_tensor[s, 1] = float(seg_dop[s].item()) / dop_sum
            seg_feat_mask[s, 1] = 1.0
    else:
        for s in active_seg_idx:
            seg_feat_mask[s, 1] = 1.0

    # seg feature #0: per-segment volume share split by block_id
    block_feat = torch.zeros(2, 1)
    block_feat_mask = torch.zeros_like(block_feat)

    block_volume = {0: 0.0, 1: 0.0}
    block_has = {0: False, 1: False}
    if "f1" in row and not pd.isna(row["f1"]):
        block_volume[0] = float(row["f1"])
        block_has[0] = True
    if "f2" in row and not pd.isna(row["f2"]):
        block_volume[1] = float(row["f2"])
        block_has[1] = True

    if volume_mode == "with":
        for b in (0, 1):
            if block_has[b]:
                block_feat[b, 0] = block_volume[b]
                block_feat_mask[b, 0] = 1.0

            segs_in_block = [
                s for s in active_seg_idx
                if int(seg_block_id[s].item()) == b
            ]
            if segs_in_block and block_has[b]:
                share = block_volume[b] / float(len(segs_in_block))
                for s in segs_in_block:
                    seg_feat_tensor[s, 0] = share
                    seg_feat_mask[s, 0] = 1.0

    elif volume_mode == "without":
        pass
    else:
        raise ValueError(f"Unknown volume_mode={volume_mode}")

    N_atom = int(len(atom_token))
    special_T = 2 + max_segments
    T = special_T + N_atom
    base_mask = torch.zeros(T, T, dtype=torch.float32)

    if float(glob_valid_mask.item()) == 0.0:
        base_mask[1, :] = float("-inf")
        base_mask[:, 1] = float("-inf")

    for s in range(max_segments):
        if float(seg_valid_mask[s].item()) == 0.0:
            pos = 2 + s
            base_mask[pos, :] = float("-inf")
            base_mask[:, pos] = float("-inf")

    labels = {c: (None if pd.isna(row[c]) else row[c]) for c in label_cols}
    
    ### Chain-level features here
    # ---- chain-level global features ---- #########################NOOOOOOOOOOOOOOOOOOOOOTE! GENERELIZATION NOT REALIZED!!!!!!!!!!!#############################
    chain_glob_feat = []
    chain_glob_mask = []
    
    # temperature
    if "glob_feat0" in row and not pd.isna(row["glob_feat0"]):
        chain_glob_feat.append(float(row["glob_feat0"]))
        chain_glob_mask.append(1)
    else:
        chain_glob_feat.append(0.0)
        chain_glob_mask.append(0)
    
    # number-average molecular weight
    if "glob_feat1" in row and not pd.isna(row["glob_feat1"]):
        chain_glob_feat.append(float(row["glob_feat1"]))
        chain_glob_mask.append(1)
    else:
        chain_glob_feat.append(0.0)
        chain_glob_mask.append(0)
    
    # coexistence flag
    if "is_coexistence" in row and not pd.isna(row["is_coexistence"]):
        chain_glob_feat.append(float(row["is_coexistence"]))
        chain_glob_mask.append(1)
    else:
        chain_glob_feat.append(0.0)
        chain_glob_mask.append(0)
    
    chain_glob_feat = torch.tensor(chain_glob_feat, dtype=torch.float32)
    chain_glob_mask = torch.tensor(chain_glob_mask, dtype=torch.float32)

    # ---- build chain index ----
    chain_block = []
    for s in range(max_segments):
        n = int(seg_dop[s].item())
        if n > 0:
            chain_block.extend([int(seg_block_id[s].item())] * n)
    
    chain_block = torch.tensor(chain_block, dtype=torch.long)
    chain_len = chain_block.numel()
    
    # ---- chain distance matrix ----
    idx = torch.arange(chain_len, dtype=torch.long)

    h = hashlib.sha256()
    h.update("|".join(smiles_list).encode())
    for v in glob_feat:
        h.update(str(v).encode())
    for s in range(max_segments):
        for f in range(MAX_LOCAL_FEATS):
            h.update(str(float(seg_feat_tensor[s, f].item())).encode())
    polymer_id = h.hexdigest()

    return {
        "polymer_id": polymer_id,
        "src_token": atom_token,
        "src_pos": src_pos,
        "atom_feat": atom_feat,
        "atom_mask": atom_mask,
        "edge_feat": edge_feat,
        "shortest_path": shortest_path,
        "degree": degree,
        "pair_type": pair_type,
        "segment_id": segment_id,
        "base_mask": base_mask,
        "glob_feat": glob_feat_tensor,
        "glob_mask": glob_mask_tensor,
        "glob_valid_mask": glob_valid_mask.unsqueeze(0),
        "seg_feat": seg_feat_tensor,
        "seg_feat_mask": seg_feat_mask,
        "seg_valid_mask": seg_valid_mask,
        "seg_dop": seg_dop,
        "seg_block_id": seg_block_id,
        "seg_smiles_id": seg_smiles_id,
        "chain_block": chain_block,
        "chain_len": chain_len,
        "chain_glob_feat": chain_glob_feat,
        "chain_glob_mask": chain_glob_mask,
        "block_feat": block_feat,
        "block_feat_mask": block_feat_mask,
        "special_T": special_T,
        "label": labels,
        "row_idx": row_idx,
        "split": "unsplit",
    }


def _resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    if args.outroot is not None:
        dataset_name = args.dataset_name or args.csv.stem
        base = Path(args.outroot) / dataset_name
        main_dir = base / "main"
        index_dir = base / "index"
        csv_dir = base / "csv"
        main_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)

        if args.output is not None:
            main_path = main_dir / args.output.name
        else:
            main_path = main_dir / "split.pkl"

        return main_path, index_dir, csv_dir

    if args.output is None:
        raise ValueError("--output must be provided when --outroot is not set.")

    main_path = args.output
    index_dir = args.output.parent
    csv_dir = args.output.parent
    index_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    return main_path, index_dir, csv_dir


def _assign_folds_to_samples(
    samples: List[Dict[str, Any]],
    df: pd.DataFrame,
    kfold: int,
    seed: int,
) -> np.ndarray:
    samples.sort(key=lambda x: int(x["row_idx"]))

    n = len(samples)
    fold_ids = np.zeros(n, dtype=np.int64)

    if "fold" in df.columns:
        for i, s in enumerate(samples):
            ridx = int(s["row_idx"])
            fold_val = int(df.loc[ridx, "fold"]) % kfold
            fold_ids[i] = fold_val
            s["split"] = f"fold{fold_val}"
    else:
        rng = random.Random(seed)
        idxs = list(range(n))
        rng.shuffle(idxs)
        for i, sidx in enumerate(idxs):
            fold_val = int(i % kfold)
            fold_ids[sidx] = fold_val
        for i, s in enumerate(samples):
            s["split"] = f"fold{int(fold_ids[i])}"

    return fold_ids


def _build_train_val_test_indices_for_fold(
    fold_ids: np.ndarray,
    k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(fold_ids))
    all_idx = np.arange(n, dtype=np.int64)

    test_mask = fold_ids == k
    test_idx = all_idx[test_mask]

    pool_idx = all_idx[~test_mask]

    rng = np.random.RandomState(seed + 1000 + k)
    pool_idx_shuffled = pool_idx.copy()
    rng.shuffle(pool_idx_shuffled)

    n_pool = int(len(pool_idx_shuffled))
    n_train = int(round(n_pool * 0.9))
    n_train = max(0, min(n_train, n_pool))

    train_idx = pool_idx_shuffled[:n_train]
    val_idx = pool_idx_shuffled[n_train:]

    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def export_periogt_style_split_fold_pkls(
    fold_ids: np.ndarray,
    kfold: int,
    seed: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(kfold):
        train_idx, val_idx, test_idx = _build_train_val_test_indices_for_fold(fold_ids, k, seed)
        split = [train_idx, val_idx, test_idx]

        out_path = out_dir / f"split_fold{k}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(split, f)

        print(
            f"[OK] PerioGT split written: {out_path} "
            f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
        )


def export_kfold_train_val_csvs(
    df: pd.DataFrame,
    samples: List[Dict[str, Any]],
    fold_ids: np.ndarray,
    kfold: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_row_idx = np.array([int(s["row_idx"]) for s in samples], dtype=np.int64)

    for k in range(kfold):
        val_mask = fold_ids == k
        train_mask = ~val_mask

        val_rows = sample_row_idx[val_mask].tolist()
        train_rows = sample_row_idx[train_mask].tolist()

        train_df = df.loc[train_rows].copy().sort_index()
        val_df = df.loc[val_rows].copy().sort_index()

        train_path = out_dir / f"train{k}.csv"
        val_path = out_dir / f"val{k}.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        print(
            f"[OK] CSV written: {train_path} (n={len(train_df)}) | "
            f"{val_path} (n={len(val_df)})"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser("CSV to polymer.pkl with full graph features")
    parser.add_argument("--task", required=True, choices=["finetune", "pretrain"], default="finetune")
    parser.add_argument("--csv", required=True, type=Path)

    parser.add_argument("--output", type=Path, default=None, help="main output pkl for finetune task")
    parser.add_argument("--outdir", type=Path, help="output dir for pretrain task")

    parser.add_argument("--smiles-prefix", default="SMILES")
    parser.add_argument("--labels", default="label", help="Comma-separated list of label columns")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--volume-mode",
        type=str,
        choices=["with", "without"],
        default="with",
        help="Whether to include volume fraction features in generated tensors.",
    )

    parser.add_argument(
        "--export-splits",
        action="store_true",
        help="export split_fold{k}.pkl and train/val csvs (finetune only)",
    )

    parser.add_argument(
        "--outroot",
        type=Path,
        default=None,
        help="root directory to organize outputs by dataset",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="dataset name under outroot (default: csv stem)",
    )

    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)

    smiles_cols = sorted(
        [c for c in df.columns if re.fullmatch(rf"{re.escape(args.smiles_prefix)}\d+", c)],
        key=lambda x: int(x[len(args.smiles_prefix):]),
    )
    if not smiles_cols:
        print("[ERROR] No SMILES columns found", file=sys.stderr)
        sys.exit(1)
    max_segments = len(smiles_cols)

    label_cols = [c.strip() for c in args.labels.split(",") if c.strip()]
    label_cols = [c for c in label_cols if c in df.columns]

    # Build a dataset-level SMILES vocabulary for stage2 token embedding.
    unique_smiles = set()
    for c in smiles_cols:
        vals = df[c].dropna().astype(str)
        vals = vals[vals != ""]
        unique_smiles.update(vals.tolist())
    smiles_vocab = {smi: i + 1 for i, smi in enumerate(sorted(unique_smiles))}

    if args.task == "finetune":
        try:
            main_path, index_dir, csv_dir = _resolve_output_paths(args)
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

        tasks = [
            (idx, row, smiles_cols, label_cols, max_segments, args.volume_mode, smiles_vocab)
            for idx, row in df.iterrows()
        ]
        samples: List[Dict[str, Any]] = []
        failed: List[Tuple[Any, Any]] = []

        with ThreadPool(args.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_row, tasks),
                total=len(tasks),
                desc="Processing rows",
            ):
                if result.get("_error"):
                    failed.append((result.get("row_idx"), result.get("failed_smi")))
                else:
                    samples.append(result)

        print(f"Processed {len(samples)}/{len(df)} rows successfully. {len(failed)} failed.")
        if failed:
            print("Failed rows:")
            for idx, smi in failed:
                print(f"  Row {idx}, SMILES = {smi}")

        fold_ids = _assign_folds_to_samples(samples, df, args.kfold, args.seed)

        header = {
            "glob_feature_count": MAX_GLOBAL_FEATS,
            "local_feature_count": MAX_LOCAL_FEATS,
            "max_segments": max_segments,
            "smiles_vocab_size": len(smiles_vocab) + 1,
            "label_names": label_cols,
            "samples": samples,
        }

        main_path.parent.mkdir(parents=True, exist_ok=True)
        with main_path.open("wb") as f:
            pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("[OK] Main finetune pkl saved to ->", main_path)

        if args.export_splits:
            export_periogt_style_split_fold_pkls(fold_ids, args.kfold, args.seed, index_dir)
            export_kfold_train_val_csvs(df, samples, fold_ids, args.kfold, csv_dir)

    else:
        random.seed(args.seed)
        if args.outdir is None:
            print("[ERROR] For pretrain task, --outdir must be provided!", file=sys.stderr)
            sys.exit(1)

        CHUNK = 2000
        outdir = Path(args.outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        current, shard_id = 0, 0
        samples: List[Dict[str, Any]] = []
        failed: List[Tuple[Any, Any]] = []

        with get_context("spawn").Pool(args.workers) as pool:
            for df_chunk in pd.read_csv(args.csv, chunksize=10000):
                tasks = [
                    (idx, row, smiles_cols, label_cols, max_segments, args.volume_mode, smiles_vocab)
                    for idx, row in df_chunk.iterrows()
                ]
                for result in tqdm(
                    pool.imap_unordered(_process_row, tasks),
                    total=len(tasks),
                    desc=f"Processing chunk {shard_id}",
                    mininterval=2,
                    smoothing=0.1,
                ):
                    if result.get("_error"):
                        failed.append((result.get("row_idx"), result.get("failed_smi")))
                        continue

                    result["split"] = f"fold{random.randrange(args.kfold)}"
                    samples.append(result)
                    current += 1

                    if current == CHUNK:
                        _dump_one(shard_id, samples, outdir, label_cols, max_segments)
                        shard_id += 1
                        current = 0
                        samples.clear()
                        gc.collect()

        if samples:
            _dump_one(shard_id, samples, outdir, label_cols, max_segments)

        print(f"[OK] {shard_id + 1} shards in total: {len(failed)} samples have failed")


if __name__ == "__main__":
    main()
