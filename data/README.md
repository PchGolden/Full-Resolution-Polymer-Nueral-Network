# Data Included in the GitHub Package

This directory contains lightweight raw CSVs, split metadata and smoke-test subsets.

Large processed pickle caches and OOF prediction/embedding caches are intentionally not included in the GitHub tree. They should be distributed as release assets if needed for fast figure reproduction.

- `raw/bcdb/`: BCDB raw CSVs and small dictionaries.
- `raw/homopolymer/`: homopolymer benchmark CSV.
- `raw/md_final1640_v2/`: final MD benchmark CSV.
- `splits/`: fold metadata and split CSVs.
- `mini_smoke/`: small subsets for sanity checks.
