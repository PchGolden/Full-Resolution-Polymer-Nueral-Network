"""Regenerate selected manuscript figures from cached/source-table inputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FIGURE_COMMANDS = {
    "bcdb": [sys.executable, str(ROOT / "scripts/figures/plot_bcdb_oof_figures.py")],
    "md": [sys.executable, str(ROOT / "scripts/figures/make_md_final1640_mechanism_figure.py")],
    "md_final1640_v2": [sys.executable, str(ROOT / "scripts/figures/make_md_final1640_mechanism_figure.py")],
    "homopolymer": [sys.executable, str(ROOT / "scripts/figures/plot_homopolymer_oof_scatter.py")],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate selected FRPN manuscript figures.")
    parser.add_argument("target", choices=sorted(FIGURE_COMMANDS), help="Figure family to regenerate")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to the underlying script")
    args = parser.parse_args()
    subprocess.check_call(FIGURE_COMMANDS[args.target] + args.extra)


if __name__ == "__main__":
    main()
