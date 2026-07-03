#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rebuild final_1640dataset.csv with an explicit fold column by reusing the
historical MD_FINAL_1600 fold assignment on a structural-signature basis.

The final CSV has the same structural rows but a different row order, so we
cannot copy folds by row index directly. Instead, we hash each row using the
structure-defining columns and map the unique signature to its original fold.

Outputs:
  - csv/with_fold.csv
  - index/fold_assignments.csv

Usage:
  python -u frpn/pipelines/md_final1640_v2/preprocess/rebuild_final1640_with_fold.py \
    --new_csv paper_results/final_1640dataset.csv \
    --old_with_fold_csv data/processed/MD/MD_FINAL_1600/csv/with_fold.csv \
    --outroot data/processed/MD/MD_FINAL1640_V2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


STRUCTURE_COLS = [
    "topology",
    "glob_feat0",
    "Index",
    "rigid_ratio",
    "polar_ratio",
    *[f"SMILES{i}" for i in range(8)],
    *[f"seg{i}_feat0" for i in range(8)],
    "chain_node_seg_id",
    "chain_edges",
    "chain_node_types",
    "chem_profile_id",
    "mix_mode",
]


def _canonical_scalar(col: str, value) -> str:
    if pd.isna(value):
        return ""
    if col in {"topology", "mix_mode"}:
        return str(value).strip()
    if col.startswith("SMILES") or col in {"chain_node_seg_id", "chain_edges", "chain_node_types"}:
        text = str(value).strip()
        if not text:
            return ""
        if col in {"chain_node_seg_id", "chain_edges", "chain_node_types"}:
            try:
                obj = json.loads(text)
                return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
            except Exception:
                return text
        return text
    try:
        return format(float(value), ".10g")
    except Exception:
        return str(value).strip()


def _row_signature(row: pd.Series, cols: Iterable[str]) -> str:
    parts = [_canonical_scalar(c, row[c]) for c in cols]
    sig = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(sig).hexdigest()


def _build_signature_fold_map(df: pd.DataFrame) -> pd.DataFrame:
    if "fold" not in df.columns:
        raise ValueError("old_with_fold_csv must contain a fold column")
    if "row_idx" in df.columns:
        raise ValueError("old_with_fold_csv must be the raw with_fold.csv, not fold_assignments.csv")

    work = df.copy()
    work["__sig__"] = work.apply(lambda r: _row_signature(r, STRUCTURE_COLS), axis=1)
    bad = work.groupby("__sig__")["fold"].nunique()
    bad = bad[bad > 1]
    if not bad.empty:
        raise RuntimeError(
            f"Old fold assignment is not signature-consistent for {len(bad)} structures; "
            "cannot reuse folds safely."
        )
    sig_map = work.groupby("__sig__", as_index=False)["fold"].first()
    return sig_map


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild final_1640dataset with reused folds")
    ap.add_argument("--new_csv", type=Path, required=True)
    ap.add_argument("--old_with_fold_csv", type=Path, required=True)
    ap.add_argument("--outroot", type=Path, required=True)
    args = ap.parse_args()

    new_df = pd.read_csv(args.new_csv)
    old_df = pd.read_csv(args.old_with_fold_csv)

    missing = [c for c in STRUCTURE_COLS if c not in new_df.columns or c not in old_df.columns]
    if missing:
        raise ValueError(f"Missing structure columns in one of the CSVs: {missing}")

    new_df = new_df.copy()
    new_df["__sig__"] = new_df.apply(lambda r: _row_signature(r, STRUCTURE_COLS), axis=1)
    sig_map = _build_signature_fold_map(old_df)

    merged = new_df.merge(sig_map, on="__sig__", how="left", validate="many_to_one")
    if merged["fold"].isna().any():
        n_missing = int(merged["fold"].isna().sum())
        raise RuntimeError(f"Failed to map fold for {n_missing} rows from the final CSV")
    merged["fold"] = merged["fold"].astype(int)

    # QC: fold distribution should match the old split counts.
    print("[QC] final fold sizes:", merged["fold"].value_counts().sort_index().to_dict())
    print("[QC] topology x fold:")
    print(pd.crosstab(merged["fold"], merged["topology"]))
    print("[QC] temperature x fold:")
    print(pd.crosstab(merged["fold"], merged["glob_feat0"]))

    outroot = args.outroot
    csv_dir = outroot / "csv"
    index_dir = outroot / "index"
    csv_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    with_fold_path = csv_dir / "with_fold.csv"
    merged.drop(columns=["__sig__"]).to_csv(with_fold_path, index=False)

    assign_df = pd.DataFrame(
        {
            "row_idx": np.arange(len(merged), dtype=np.int64),
            "group_id": merged["__sig__"].astype(str).tolist(),
            "stratum": merged.apply(lambda r: f"{r['topology']}|T{int(float(r['glob_feat0']))}", axis=1).tolist(),
            "fold": merged["fold"].astype(int).tolist(),
        }
    )
    assign_path = index_dir / "fold_assignments.csv"
    assign_df.to_csv(assign_path, index=False)

    print(f"[OK] wrote: {with_fold_path} (n={len(merged)})")
    print(f"[OK] wrote: {assign_path} (n={len(assign_df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
