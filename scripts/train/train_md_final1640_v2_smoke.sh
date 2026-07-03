#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo_root" python - <<'PY'
from frpn.cli._run_pipeline import pipeline_path
p = pipeline_path('md_final1640_v2')
assert (p / 'main.py').exists(), p
assert (p / 'analysis' / 'run_md_final1640_mechanism_reproduction.py').exists(), p
print('MD_FINAL1640_v2 pipeline layout smoke test passed.')
PY
