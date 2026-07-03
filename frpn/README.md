# FRPN Source Package

This package contains the public-release source tree for FRPN: a full-resolution polymer representation framework for cross-scale interaction learning.

## Layout

- `cli/`: command-line wrappers.
- `pipelines/bcdb/`: BCDB and homopolymer-style training/evaluation pipeline.
- `pipelines/md_final1640_v2/`: MD benchmark training/evaluation and mechanism-analysis pipeline.

The two pipelines are kept separate because their chain reconstruction, label handling and benchmark-specific options differ. The layout replaces the original `src/` and `src_MD/` split with explicit pipeline names, while preserving model logic, batching, normalization, metrics and training behavior.

## Entry Points

- `python -m frpn.cli.train_bcdb`
- `python -m frpn.cli.train_homopolymer`
- `python -m frpn.cli.train_md`
- `python -m frpn.cli.make_figures {bcdb,homopolymer,md_final1640_v2}`
- `python -m frpn.cli.export_oof`

## Merge Boundary

BCDB chain graphs are reconstructed from polymer descriptions, SMILES/BigSMILES, degree of polymerization, block fraction and volume features. MD chain graphs are read from explicit dataset fields such as chain node segment IDs and chain edges. These differences are intentionally represented as separate pipelines rather than hidden behind a fragile abstraction.
