"""Helpers for invoking FRPN pipeline entry points without changing behavior."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PIPELINES_ROOT = PACKAGE_ROOT / "pipelines"

PIPELINE_ALIASES = {
    "bcdb": "bcdb",
    "homopolymer": "bcdb",
    "md": "md_final1640_v2",
    "md_final1640_v2": "md_final1640_v2",
}


def pipeline_path(name: str) -> Path:
    """Return the on-disk directory for a named FRPN pipeline."""

    key = PIPELINE_ALIASES.get(name, name)
    path = PIPELINES_ROOT / key
    if not path.exists():
        raise FileNotFoundError(f"Unknown FRPN pipeline '{name}': {path}")
    return path


def run_pipeline_script(pipeline: str, relative_script: str = "main.py") -> None:
    """Run a pipeline script as if it were called from its original tree.

    The training scripts still use local imports such as ``from models...``.
    We therefore prepend the selected pipeline directory to ``sys.path``. This
    preserves the original behavior while exposing a cleaner package layout.
    """

    root = pipeline_path(pipeline)
    script_path = root / relative_script
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")

    sys.path.insert(0, str(root))
    runpy.run_path(str(script_path), run_name="__main__")
