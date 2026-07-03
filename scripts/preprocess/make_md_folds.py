#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a leakage-safe 5-fold split for the final MD regression dataset by:
- grouping by polymer structure/topology (excluding temperature + labels)
- stratifying by (topology, temperature) buckets
- assigning groups to folds via deterministic per-bucket round-robin

Outputs:
- with_fold.csv (original rows + fold column)
- fold_assignments.csv (row_idx, group_id, stratum, fold)

Example:
  python -u frpn/pipelines/md_final1640_v2/preprocess/make_folds_md_final.py \
    --csv data/raw/merged_534_orth800_380_eqnpt_extra_labels.csv \
    --outroot data/processed/MD/MD_FINAL_1600 \
    --seed 42 \
    --kfold 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _canonical_json(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return ""
    try:
        obj = json.loads(s)
    except Exception:
        return s
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return s


def _canonical_num(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    try:
        return format(float(x), ".10g")
    except Exception:
        return str(x)


def _hash_group(parts: List[str]) -> str:
    sig = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(sig).hexdigest()


def _build_group_and_stratum(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    smiles_cols = [c for c in [f"SMILES{i}" for i in range(8)] if c in df.columns]
    seg_feat_cols = [c for c in [f"seg{i}_feat0" for i in range(8)] if c in df.columns]

    required_cols = ["topology", "glob_feat0", "mix_mode", "chain_node_seg_id", "chain_edges"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    group_ids: List[str] = []
    strata: List[str] = []

    for _, row in df.iterrows():
        topology = "" if pd.isna(row["topology"]) else str(row["topology"])
        temp = "" if pd.isna(row["glob_feat0"]) else str(int(float(row["glob_feat0"])))
        mix_mode = "" if pd.isna(row["mix_mode"]) else str(row["mix_mode"])

        stratum = f"{topology}|T{temp}"
        strata.append(stratum)

        parts: List[str] = [topology, mix_mode]
        for c in smiles_cols:
            v = "" if pd.isna(row[c]) else str(row[c]).strip()
            parts.append(v)
        for c in seg_feat_cols:
            parts.append(_canonical_num(row[c]))

        parts.append(_canonical_json(row["chain_node_seg_id"]))
        parts.append(_canonical_json(row["chain_edges"]))

        group_ids.append(_hash_group(parts))

    return pd.Series(group_ids, name="group_id"), pd.Series(strata, name="stratum")


def _assign_folds_round_robin(
    df: pd.DataFrame,
    group_col: str,
    stratum_col: str,
    kfold: int,
    seed: int,
) -> Dict[str, int]:
    rng = random.Random(int(seed))

    group_sizes = df.groupby(group_col).size().to_dict()
    group_to_stratum = df.groupby(group_col)[stratum_col].first().to_dict()

    stratum_to_groups: Dict[str, List[str]] = {}
    for gid, st in group_to_stratum.items():
        stratum_to_groups.setdefault(st, []).append(gid)

    group_to_fold: Dict[str, int] = {}
    for st in sorted(stratum_to_groups.keys()):
        gids = list(stratum_to_groups[st])
        rng.shuffle(gids)

        start = rng.randrange(kfold) if kfold > 0 else 0
        for j, gid in enumerate(gids):
            group_to_fold[gid] = int((start + j) % kfold)

        # Best-effort sanity output
        n_groups = len(gids)
        n_rows = int(sum(group_sizes[g] for g in gids))
        print(f"[Split] stratum={st} groups={n_groups} rows={n_rows}")

    if len(group_to_fold) != len(group_to_stratum):
        raise RuntimeError("Internal error: group_to_fold size mismatch")

    return group_to_fold


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser("Make leakage-safe folds for final MD dataset")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--outroot", type=Path, required=True)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    group_id, stratum = _build_group_and_stratum(df)
    df = df.copy()
    df["group_id"] = group_id
    df["stratum"] = stratum

    group_to_fold = _assign_folds_round_robin(
        df,
        group_col="group_id",
        stratum_col="stratum",
        kfold=int(args.kfold),
        seed=int(args.seed),
    )
    df["fold"] = df["group_id"].map(group_to_fold).astype(int)

    # Leakage check: each group_id must map to exactly one fold
    leaks = df.groupby("group_id")["fold"].nunique()
    bad = leaks[leaks > 1]
    if len(bad) > 0:
        raise RuntimeError(f"Leakage detected: {len(bad)} group_id(s) appear in multiple folds")

    # QC summaries
    print("[QC] fold sizes:", df["fold"].value_counts().sort_index().to_dict())
    if "glob_feat0" in df.columns:
        print("[QC] temp by fold:")
        print(pd.crosstab(df["fold"], df["glob_feat0"]))
    if "topology" in df.columns:
        print("[QC] topology by fold:")
        print(pd.crosstab(df["fold"], df["topology"]))
        print("[QC] stratum(topology|T) by fold:")
        print(pd.crosstab(df["fold"], df["stratum"]))

    outroot = args.outroot
    csv_dir = outroot / "csv"
    index_dir = outroot / "index"
    csv_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    with_fold_path = csv_dir / "with_fold.csv"
    df.drop(columns=["group_id", "stratum"]).to_csv(with_fold_path, index=False)

    assign_path = index_dir / "fold_assignments.csv"
    assign_df = pd.DataFrame(
        {
            "row_idx": np.arange(len(df), dtype=np.int64),
            "group_id": df["group_id"].astype(str).tolist(),
            "stratum": df["stratum"].astype(str).tolist(),
            "fold": df["fold"].astype(int).tolist(),
        }
    )
    assign_df.to_csv(assign_path, index=False)

    print(f"[OK] with_fold.csv -> {with_fold_path} (n={len(df)})")
    print(f"[OK] fold_assignments.csv -> {assign_path} (n={len(assign_df)})")


if __name__ == "__main__":
    main()

