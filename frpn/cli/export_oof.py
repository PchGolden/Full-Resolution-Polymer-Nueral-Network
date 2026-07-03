"""Documented placeholder for OOF export in the publication package.

OOF predictions and embeddings used by the manuscript are staged under
``results/oof_predictions`` and ``results/oof_embeddings``. Re-exporting them
from checkpoints requires external checkpoint assets and the curated cluster job
references under ``configs/``.
"""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    print("OOF caches are already staged for figure/table reproduction:")
    print(f"  {root / 'results/oof_predictions'}")
    print(f"  {root / 'results/oof_embeddings'}")
    print("For checkpoint-based re-export, first populate external assets listed in:")
    print(f"  {root / 'checkpoints/checkpoints_manifest.csv'}")
    print("Then use the curated job/config references under configs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
