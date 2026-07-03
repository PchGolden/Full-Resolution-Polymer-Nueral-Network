# preprocessing/preprocessing_polymer.py
# Combined preprocessing script supporting multiple SMILES segments and full graph features.

import random
import argparse
import hashlib
import json
import sys
import pickle
import gc
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional

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
MAX_SEGMENTS = 8
MAX_LOCAL_FEATS = 2
MAX_GLOBAL_FEATS = 2
SEG_SMILES_VOCAB_SIZE = 20000

#### MULTI-CONF CONFIGS ####
NUM_CONFS = 4
MAX_TRIES = NUM_CONFS * 2


def _dump_one(i: int, samples: List[Dict[str, Any]], outdir: Path, label_cols: List[str]) -> None:
    header = {
        "glob_feature_count": MAX_GLOBAL_FEATS,
        "local_feature_count": MAX_LOCAL_FEATS,
        "max_segments": MAX_SEGMENTS,
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


def _embed_and_strip_h(smiles: str) -> Optional[Tuple[Chem.Mol, np.ndarray]]:
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
    no_h_3d = None  # type: Optional[Chem.Mol]

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


def _smiles_to_vocab_id(
    smiles: str,
    smiles_vocab: Optional[Dict[str, int]] = None,
    vocab_size: int = SEG_SMILES_VOCAB_SIZE,
) -> int:
    # Preferred path: dataset-specific dense ids in [1, N].
    if smiles_vocab is not None:
        return int(smiles_vocab.get(smiles, 0))

    # Backward-compatible fallback: hash to [1, vocab_size-1].
    if not smiles:
        return 0
    h = hashlib.sha256(smiles.encode("utf-8")).digest()
    raw = int.from_bytes(h[:8], byteorder="big", signed=False)
    if vocab_size <= 1:
        return 0
    return int(raw % (vocab_size - 1)) + 1


def _process_row(args: tuple) -> Dict[str, Any]:
    if len(args) >= 5:
        row_idx, row, smiles_cols, label_cols, smiles_vocab = args
    else:
        row_idx, row, smiles_cols, label_cols = args
        smiles_vocab = None
    # Keep the *original* segment index from the column suffix.
    # Important when some SMILES cells are empty (e.g., SMILES0 + SMILES2 exist,
    # but SMILES1 is empty): we must not renumber segments.
    smiles_items: List[Tuple[int, str]] = []
    for item in smiles_cols:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            seg_i, col = int(item[0]), str(item[1])
        else:
            col = str(item)
            m = re.search(r"(\d+)$", col)
            if m is None:
                continue
            seg_i = int(m.group(1))

        if col not in row:
            continue

        v = row[col]
        if pd.isna(v) or v == "":
            continue
        smiles_items.append((int(seg_i), str(v)))

    if not smiles_items:
        return {"_error": True, "row_idx": row_idx, "failed_smi": None}

    segment_ids: List[int] = []
    coords_list: List[np.ndarray] = []
    node_feats_all: List[np.ndarray] = []
    edge_feats_all: List[torch.Tensor] = []
    sp_all: List[torch.Tensor] = []
    degree_all: List[torch.Tensor] = []
    atom_mask_all: List[torch.Tensor] = []
    atom_counts: List[int] = []
    atomic_nums_all: List[torch.Tensor] = []

    for seg_i, smi in smiles_items:
        res = _embed_and_strip_h(smi)
        if res is None:
            return {"_error": True, "row_idx": row_idx, "failed_smi": smi}
        mol, coords_arr = res
        node_attr, edge_index, edge_attr, proc_mol, atomic_nums = build_initial_graph(mol)
        atomic_nums_all.append(torch.from_numpy(atomic_nums).long())
        feat = build_graph_features(node_attr, edge_index, edge_attr, proc_mol)

        N = node_attr.shape[0]
        atom_counts.append(N)
        segment_ids.extend([seg_i] * N)

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
    seg_smiles_id = torch.zeros(MAX_SEGMENTS, dtype=torch.long)
    for seg_i, smi in smiles_items:
        if 0 <= int(seg_i) < MAX_SEGMENTS:
            seg_smiles_id[int(seg_i)] = int(_smiles_to_vocab_id(str(smi), smiles_vocab=smiles_vocab))

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

    local_feat: List[List[Any]] = [[None] * MAX_LOCAL_FEATS for _ in range(MAX_SEGMENTS)]
    for s in range(MAX_SEGMENTS):
        for f in range(MAX_LOCAL_FEATS):
            col = f"seg{s}_feat{f}"
            if col in row and not pd.isna(row[col]):
                local_feat[s][f] = row[col]

    glob_feat_tensor, glob_mask_tensor = _build_numeric_tensor(glob_feat)
    glob_valid_mask = (glob_mask_tensor.sum() > 0).float()

    seg_feat_tensor = torch.zeros(MAX_SEGMENTS, MAX_LOCAL_FEATS)
    seg_feat_mask = torch.zeros_like(seg_feat_tensor)

    for s in range(MAX_SEGMENTS):
        for f in range(MAX_LOCAL_FEATS):
            v = local_feat[s][f]
            if v is not None:
                seg_feat_tensor[s, f] = float(v)
                seg_feat_mask[s, f] = 1

    seg_valid_mask = (seg_feat_mask.sum(-1) > 0).float()

    N_atom = int(len(atom_token))
    special_T = 2 + MAX_SEGMENTS
    T = special_T + N_atom
    base_mask = torch.zeros(T, T, dtype=torch.float32)

    if float(glob_valid_mask.item()) == 0.0:
        base_mask[1, :] = float("-inf")
        base_mask[:, 1] = float("-inf")

    for s in range(MAX_SEGMENTS):
        if float(seg_valid_mask[s].item()) == 0.0:
            pos = 2 + s
            base_mask[pos, :] = float("-inf")
            base_mask[:, pos] = float("-inf")

    labels = {c: (None if pd.isna(row[c]) else row[c]) for c in label_cols}
    if label_cols:
        missing = [c for c in label_cols if labels.get(c, None) is None]
        if missing:
            return {"_error": True, "row_idx": row_idx, "failed_smi": f"<missing_label:{','.join(missing)}>"}

    # ------------------------------------------------------------------
    # Stage-2 topology-driven fields (new full-resolution path)
    # ------------------------------------------------------------------
    chain_node_seg_id = None
    chain_topo_dist = None

    def _build_topo_dist(num_nodes: int, edges: List[List[int]]) -> torch.Tensor:
        # Unweighted graph shortest-path distances by BFS from each node.
        if num_nodes <= 0:
            return torch.zeros(0, 0, dtype=torch.long)

        adj = [[] for _ in range(num_nodes)]
        for e in edges:
            if not isinstance(e, (list, tuple)) or len(e) != 2:
                continue
            u, v = int(e[0]), int(e[1])
            if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
                continue
            adj[u].append(v)
            adj[v].append(u)

        INF = num_nodes + 1
        dist_mat = [[INF] * num_nodes for _ in range(num_nodes)]
        for s in range(num_nodes):
            dist_mat[s][s] = 0
            q = [s]
            head = 0
            while head < len(q):
                x = q[head]
                head += 1
                dx = dist_mat[s][x]
                for y in adj[x]:
                    if dist_mat[s][y] == INF:
                        dist_mat[s][y] = dx + 1
                        q.append(y)

        return torch.tensor(dist_mat, dtype=torch.long)

    if "chain_node_seg_id" in row and pd.notna(row["chain_node_seg_id"]) and str(row["chain_node_seg_id"]).strip() != "":
        try:
            chain_node_seg_id_list = json.loads(str(row["chain_node_seg_id"]))
            chain_node_seg_id = torch.tensor(chain_node_seg_id_list, dtype=torch.long)
        except Exception:
            chain_node_seg_id = None

    if chain_node_seg_id is not None and "chain_edges" in row and pd.notna(row["chain_edges"]) and str(row["chain_edges"]).strip() != "":
        try:
            edges = json.loads(str(row["chain_edges"]))
            chain_topo_dist = _build_topo_dist(int(chain_node_seg_id.numel()), edges)
        except Exception:
            chain_topo_dist = None
    
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

    # ---- block-level features ----
    # For the new topology-driven workflow, block partitioning may be undefined.
    # Keep blocks as an *empty* padded tensor by default (Bmax=0), which the model supports.
    # If you later provide explicit block features, populate them here.
    block_feat = torch.zeros(0, 1, dtype=torch.float32)        # [Bmax=0, F=1]
    block_feat_mask = torch.zeros_like(block_feat)

    
    # ---- seg dop & block id ----
    seg_dop = torch.zeros(MAX_SEGMENTS, dtype=torch.long)
    seg_block_id = torch.zeros(MAX_SEGMENTS, dtype=torch.long)
    
    for s in range(MAX_SEGMENTS):
        dop_col = f"seg{s}_dop_raw"
        blk_col = f"seg{s}_block_id"
    
        if dop_col in row and not pd.isna(row[dop_col]):
            seg_dop[s] = int(row[dop_col])
    
        if blk_col in row and not pd.isna(row[blk_col]):
            seg_block_id[s] = int(row[blk_col])
            
    # ---- legacy chain index (kept for backward compatibility) ----
    chain_block = []
    for s in range(MAX_SEGMENTS):
        n = int(seg_dop[s].item())
        if n > 0:
            chain_block.extend([int(seg_block_id[s].item())] * n)
    chain_block = torch.tensor(chain_block, dtype=torch.long)

    # Prefer topology-driven node count when provided
    if chain_node_seg_id is not None:
        chain_len = int(chain_node_seg_id.numel())
    else:
        chain_len = int(chain_block.numel())

    idx = torch.arange(chain_len, dtype=torch.long)
    chain_dist = torch.abs(idx[:, None] - idx[None, :])


    h = hashlib.sha256()
    # Include segment indices to avoid collisions when some segment slots are empty.
    h.update("|".join([f"{seg_i}:{smi}" for seg_i, smi in smiles_items]).encode())
    for v in glob_feat:
        h.update(str(v).encode())
    for s in range(MAX_SEGMENTS):
        for f in range(MAX_LOCAL_FEATS):
            h.update(str(local_feat[s][f]).encode())
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
        "seg_smiles_id": seg_smiles_id,
        "seg_dop": seg_dop,
        "seg_block_id": seg_block_id,
        "chain_block": chain_block,
        "chain_len": chain_len,
        "chain_node_seg_id": chain_node_seg_id,
        "chain_topo_dist": chain_topo_dist,
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
    fold_source: str = "auto",
) -> np.ndarray:
    samples.sort(key=lambda x: int(x["row_idx"]))

    n = len(samples)
    fold_ids = np.zeros(n, dtype=np.int64)

    if fold_source not in {"auto", "topology", "column", "random", "stratified_topology_group"}:
        raise ValueError(f"Unknown fold_source={fold_source}")

    if fold_source == "stratified_topology_group":
        if "topology" not in df.columns:
            raise ValueError("fold_source=stratified_topology_group requires df['topology'] column")

        try:
            from sklearn.model_selection import StratifiedGroupKFold
        except Exception:
            StratifiedGroupKFold = None

        y = []
        groups = []
        gid_to_sample_idxs: Dict[str, List[int]] = {}
        for i, s in enumerate(samples):
            ridx = int(s["row_idx"])
            topo = str(df.loc[ridx, "topology"])
            gid = str(s.get("polymer_id", ""))
            y.append(topo)
            groups.append(gid)
            gid_to_sample_idxs.setdefault(gid, []).append(i)

        if StratifiedGroupKFold is not None:
            splitter = StratifiedGroupKFold(
                n_splits=kfold,
                shuffle=True,
                random_state=seed,
            )
            for fold_val, (_, test_idx) in enumerate(splitter.split(np.zeros(n), y, groups)):
                fold_ids[test_idx] = int(fold_val)
            split_mode = "sklearn"
        else:
            # Fallback: per-topology group shuffle + round-robin into folds (no sklearn version dependency).
            rng = random.Random(seed)
            gid_to_topo: Dict[str, str] = {}
            for gid, idxs in gid_to_sample_idxs.items():
                gid_to_topo[gid] = y[idxs[0]]

            topo_to_gids: Dict[str, List[str]] = {}
            for gid, topo in gid_to_topo.items():
                topo_to_gids.setdefault(topo, []).append(gid)

            gid_to_fold: Dict[str, int] = {}
            for topo in sorted(topo_to_gids.keys()):
                gids = topo_to_gids[topo]
                rng.shuffle(gids)
                start = rng.randrange(kfold)
                for j, gid in enumerate(gids):
                    gid_to_fold[gid] = int((start + j) % kfold)

            for i, gid in enumerate(groups):
                fold_ids[i] = int(gid_to_fold.get(gid, 0))
            split_mode = "manual"

        for i, s in enumerate(samples):
            s["split"] = f"fold{int(fold_ids[i])}"

        uniq_topo = sorted(set(y))
        topo_counts = {t: 0 for t in uniq_topo}
        for t in y:
            topo_counts[str(t)] += 1
        print(f"[Split] stratified_topology_group mode={split_mode} topo_counts={topo_counts}")
        return fold_ids

    use_topology = False
    use_column = False
    if fold_source == "topology":
        use_topology = True
    elif fold_source == "column":
        use_column = True
    elif fold_source == "auto":
        if "topology" in df.columns:
            topo_vals = [str(v) for v in df["topology"].dropna().tolist()]
            uniq_topo = sorted(set(topo_vals))
            if len(uniq_topo) == kfold:
                use_topology = True
        if (not use_topology) and ("fold" in df.columns):
            use_column = True

    if use_topology:
        topo_vals = [str(v) for v in df["topology"].dropna().tolist()]
        uniq_topo = sorted(set(topo_vals))
        if len(uniq_topo) != kfold:
            raise ValueError(
                f"Topology fold mode requires kfold==#unique(topology): got kfold={kfold}, unique={len(uniq_topo)}"
            )
        topo2fold = {topo: i for i, topo in enumerate(uniq_topo)}
        print(f"[Split] topology->fold mapping: {topo2fold}")

        for i, s in enumerate(samples):
            ridx = int(s["row_idx"])
            topo = str(df.loc[ridx, "topology"])
            fold_val = int(topo2fold[topo])
            fold_ids[i] = fold_val
            s["split"] = f"fold{fold_val}"
    elif use_column:
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


def _copy_splits_from_existing_pkl(samples: List[Dict[str, Any]], pkl_path: Path, kfold: int) -> np.ndarray:
    header = pickle.load(open(pkl_path, "rb"))
    old_samples = header.get("samples", [])
    if not isinstance(old_samples, list) or not old_samples:
        raise ValueError(f"--copy-splits-from-pkl invalid: no samples in {pkl_path}")

    pid_to_split: Dict[str, str] = {}
    for s in old_samples:
        pid = str(s.get("polymer_id", ""))
        sp = str(s.get("split", ""))
        if not pid or not sp:
            continue
        if pid in pid_to_split and pid_to_split[pid] != sp:
            raise ValueError(f"polymer_id {pid} has multiple splits in {pkl_path}: {pid_to_split[pid]} vs {sp}")
        pid_to_split[pid] = sp

    if not pid_to_split:
        raise ValueError(f"--copy-splits-from-pkl invalid: missing polymer_id/split in {pkl_path}")

    fold_ids = np.zeros(len(samples), dtype=np.int64)
    missing = []
    for i, s in enumerate(samples):
        pid = str(s.get("polymer_id", ""))
        sp = pid_to_split.get(pid, None)
        if sp is None:
            missing.append(pid)
            continue
        s["split"] = sp
        if sp.startswith("fold"):
            try:
                fold_ids[i] = int(sp.replace("fold", ""))
            except Exception:
                fold_ids[i] = 0

    if missing:
        uniq = sorted(set(missing))
        raise ValueError(
            f"--copy-splits-from-pkl could not find {len(uniq)} polymer_id(s) in {pkl_path}. "
            f"Examples: {uniq[:10]}"
        )

    # Sanity: ensure fold ids are within [0, kfold)
    bad = [int(v) for v in np.unique(fold_ids) if int(v) < 0 or int(v) >= int(kfold)]
    if bad:
        raise ValueError(f"Invalid fold ids after copy: {bad} (kfold={kfold})")

    counts = {f"fold{k}": int((fold_ids == k).sum()) for k in range(int(kfold))}
    print(f"[Split] copied from {pkl_path} fold_counts={counts}")
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser("CSV to polymer.pkl with full graph features")
    parser.add_argument("--task", required=True, choices=["finetune", "pretrain"], default="finetune")
    parser.add_argument("--csv", required=True, type=Path)

    parser.add_argument("--output", type=Path, default=None, help="main output pkl for finetune task")
    parser.add_argument("--outdir", type=Path, help="output dir for pretrain task")

    parser.add_argument("--smiles-prefix", default="SMILES")
    parser.add_argument("--labels", default="label", help="Comma-separated list of label columns")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument(
        "--fold-source",
        type=str,
        default="stratified_topology_group",
        choices=["stratified_topology_group", "auto", "topology", "column", "random"],
        help=(
            "How to assign fold ids: stratified_topology_group (StratifiedGroupKFold by topology, grouped by polymer_id), "
            "topology (leave-one-topology-out), column (use df['fold']), random, or auto."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-splits-from-pkl",
        type=Path,
        default=None,
        help="(finetune only) Copy split (foldk) assignment by polymer_id from an existing processed pkl.",
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

    smiles_cols = [
        f"{args.smiles_prefix}{i}"
        for i in range(MAX_SEGMENTS)
        if f"{args.smiles_prefix}{i}" in df.columns
    ]
    if not smiles_cols:
        print("[ERROR] No SMILES columns found", file=sys.stderr)
        sys.exit(1)

    label_cols = [c.strip() for c in args.labels.split(",") if c.strip()]
    label_cols = [c for c in label_cols if c in df.columns]

    # Build dataset-specific dense SMILES vocabulary for stage-2-only ids.
    unique_smiles: List[str] = []
    seen_smiles = set()
    for col in smiles_cols:
        for v in df[col].dropna().astype(str).tolist():
            if v == "":
                continue
            if v not in seen_smiles:
                seen_smiles.add(v)
                unique_smiles.append(v)
    unique_smiles = sorted(unique_smiles)
    smiles_vocab = {smi: i + 1 for i, smi in enumerate(unique_smiles)}
    print(f"[INFO] seg_smiles vocab size (non-pad) = {len(smiles_vocab)}")

    if args.task == "finetune":
        try:
            main_path, index_dir, csv_dir = _resolve_output_paths(args)
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

        tasks = [(idx, row, smiles_cols, label_cols, smiles_vocab) for idx, row in df.iterrows()]
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

        if args.copy_splits_from_pkl is not None:
            fold_ids = _copy_splits_from_existing_pkl(samples, args.copy_splits_from_pkl, args.kfold)
        else:
            fold_ids = _assign_folds_to_samples(samples, df, args.kfold, args.seed, args.fold_source)

        header = {
            "glob_feature_count": MAX_GLOBAL_FEATS,
            "local_feature_count": MAX_LOCAL_FEATS,
            "max_segments": MAX_SEGMENTS,
            "label_names": label_cols,
            "seg_smiles_vocab_size": int(len(smiles_vocab) + 1),
            "seg_smiles_vocab": smiles_vocab,
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
                tasks = [(idx, row, smiles_cols, label_cols, smiles_vocab) for idx, row in df_chunk.iterrows()]
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
                        _dump_one(shard_id, samples, outdir, label_cols)
                        shard_id += 1
                        current = 0
                        samples.clear()
                        gc.collect()

        if samples:
            _dump_one(shard_id, samples, outdir, label_cols)

        print(f"[OK] {shard_id + 1} shards in total: {len(failed)} samples have failed")


if __name__ == "__main__":
    main()
