#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHONPATH="$repo_root/frpn/pipelines/bcdb:${PYTHONPATH:-}" python "$repo_root/frpn/pipelines/bcdb/main.py" "$@"
