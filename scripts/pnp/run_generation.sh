#!/usr/bin/env bash
# Shell boundary: resolve the interpreter/repository and invoke the Python CLI.
# All experiment logic and configuration validation live in src/pnp/run_experiment.py.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CONFIG.json [--dry-run]" >&2
  exit 2
fi

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${MOLMOSPACES_PYTHON:-python3}
command=("$python_bin" "$repository_root/src/pnp/run_experiment.py" --config "$1")
if [[ $# -eq 2 ]]; then
  [[ "$2" == "--dry-run" ]] || { echo "unsupported option: $2" >&2; exit 2; }
  command+=("$2")
fi
exec "${command[@]}"
