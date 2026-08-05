#!/usr/bin/env bash
set -euo pipefail
# Compatibility wrapper; new experiments should call run_generation.py.
ROOT="${MOLMOSPACES_ROOT:?Set MOLMOSPACES_ROOT}"
WORK="${MOLMOSPACES_PNP_WORKDIR:-$ROOT/runtime/fixedpool_potato_bowl_1716_source}"
PY="${MOLMOSPACES_PYTHON:-$ROOT/.venv/bin/python}"
exec "$PY" "$ROOT/src/pnp/run_generation.py" \
  --root "$ROOT" --python "$PY" --work "$WORK" --mode per-subtask \
  --source-hdf5 "${PNP_SOURCE_HDF5:-$WORK/artifacts/seeds/robomimic_pnp_fixedbase_17unique_replayed.hdf5}" \
  --target-manifest "${PNP_TARGET_MANIFEST:-$WORK/artifacts/seeds/pnp_seed_manifest_fixedpool.json}" \
  --target-success "${TARGET_SUCCESS:-100}" --max-attempts "${MAX_ATTEMPTS:-240}" \
  --run-label "${RUN_LABEL:-fixedbase17_per_subtask}"
