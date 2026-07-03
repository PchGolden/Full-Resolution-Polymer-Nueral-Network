"""CLI wrapper for BCDB FRPN training."""

from frpn.cli._run_pipeline import run_pipeline_script


if __name__ == "__main__":
    run_pipeline_script("bcdb", "main.py")
