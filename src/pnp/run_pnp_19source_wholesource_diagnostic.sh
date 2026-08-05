#!/usr/bin/env bash
set -euo pipefail
ROOT="${MOLMOSPACES_ROOT:?Set MOLMOSPACES_ROOT}"
WORK="${MOLMOSPACES_PNP_WORKDIR:-$ROOT/runtime/fixedbase_potato_bowl_1716_20260802}"
PY="${MOLMOSPACES_PYTHON:-/mnt/vqa/miniconda3/envs/molmospaces-integration/bin/python}"
exec "$PY" "$ROOT/src/pnp/run_generation.py" \
  --root "$ROOT" --python "$PY" --work "$WORK" --mode whole-source --diagnostic \
  --source-count "${SOURCE_COUNT:-19}" --source-hdf5 "${PNP_SOURCE_HDF5:?Set PNP_SOURCE_HDF5}" \
  --target-manifest "${PNP_TARGET_MANIFEST:-$WORK/artifacts/seeds/pnp_target_manifest_fixedbase_1716_100.json}" \
  --target-success "${TARGET_SUCCESS:-100}" --target-start "${TARGET_START:-0}" \
  --target-end "${TARGET_END:--1}" --rng-seed-base "${RNG_SEED_BASE:-20000}" \
  --run-label "${RUN_LABEL:-quarantined_repair_wholesource}"
