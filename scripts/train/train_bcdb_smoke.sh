#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo_root" python - <<'PY'
from frpn.cli._run_pipeline import pipeline_path
p = pipeline_path('bcdb')
assert (p / 'main.py').exists(), p
assert (p / 'models' / 'multi_mol_model.py').exists(), p
print('BCDB pipeline layout smoke test passed.')
PY
