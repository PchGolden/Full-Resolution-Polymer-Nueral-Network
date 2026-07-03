# FRPN

FRPN is a full-resolution polymer representation framework for cross-scale interaction learning. It organizes monomer-level chemical semantics into polymer-scale structural context so that the model can reason across chemistry, sequence/order, topology and global conditions.

This repository is the lightweight GitHub release package. Large trained checkpoints and optional cached outputs are stored separately under `release_assets/` or should be uploaded as release assets.

## Repository Contents

- `frpn/`: source package with pipeline-compatible BCDB/homopolymer and MD entry points.
- `scripts/`: curated training, evaluation, preprocessing and figure-generation scripts.
- `scripts/configs/`: cluster/job references used for the manuscript runs.
- `data/raw/`: lightweight raw benchmark CSVs.
- `data/splits/`: fold/split metadata in CSV form.
- `data/mini_smoke/`: tiny inputs for environment and parser smoke tests.
- `results/summaries/`: compact paper-summary tables.
- `results/figure_source_tables/`: source tables for the MD mechanism figure.
- `docs/`: figure/table map, checkpoint manifest, code/data availability draft and source-merge notes.

## Large Assets

Checkpoint files are not stored inside this GitHub tree. They have been copied as real files to:

```text
../release_assets/checkpoints/
```

The complete checksum manifest is:

```text
docs/checkpoint_manifest.csv
```

Before public release, upload `release_assets/checkpoints/` to Zenodo, OSF, GitHub Release assets or another archival service, then add the final download URLs to the manifest.

## Minimal Setup

```bash
mamba env create -f environment.yml
mamba activate bcdb-minimal
# or install the minimal pip requirements
pip install -r requirements.txt
```

## Smoke Tests

```bash
python -m frpn.cli.export_oof
bash scripts/train/train_bcdb_smoke.sh --help
bash scripts/train/train_md_final1640_v2_smoke.sh --help
```

Full training/evaluation requires the checkpoint assets listed in `docs/checkpoint_manifest.csv`.

## Reproducing Manuscript Results

- BCDB summaries: `results/summaries/bcdb_*.csv`.
- Homopolymer summaries: `results/summaries/homopolymer_summary_model_label.csv`.
- MD benchmark summaries: `results/summaries/md_final1640_v2_*.csv`.
- MD mechanism source tables: `results/figure_source_tables/md_final1640_v2/`.

The MD additive-null analysis should be interpreted as a functional diagnostic, not a causal proof. It supports the claim that FRPN retains complementary interaction-aligned residual information after additive chemistry, topology, ordering and temperature effects are removed.
