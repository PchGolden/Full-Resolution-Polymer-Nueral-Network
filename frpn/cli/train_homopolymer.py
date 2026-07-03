"""CLI wrapper for homopolymer runs using the BCDB-style pipeline."""

from frpn.cli._run_pipeline import run_pipeline_script


if __name__ == "__main__":
    run_pipeline_script("homopolymer", "main.py")
