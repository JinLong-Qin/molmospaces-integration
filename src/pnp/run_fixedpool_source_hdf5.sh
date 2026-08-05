#!/usr/bin/env bash
set -euo pipefail
ROOT="${MOLMOSPACES_ROOT:?Set MOLMOSPACES_ROOT}"
WORK="${MOLMOSPACES_PNP_WORKDIR:?Set MOLMOSPACES_PNP_WORKDIR}"
PY="${MOLMOSPACES_PYTHON:-$ROOT/.venv/bin/python}"
DATAGEN_ROOT="${1:?usage: $0 DATAGEN_ROOT CANDIDATE_COUNT}"
COUNT="${2:-43}"
HOUSE_ID="${PNP_FIXEDPOOL_HOUSE_ID:-1716}"
RUN_PREFIX="${PNP_FIXEDPOOL_RUN_NAME_PREFIX:-potato_bowl_${HOUSE_ID}_seed}"
exec "$PY" "$ROOT/src/pnp/run_source_hdf5_pipeline.py" \
  --root "$ROOT" --datagen-root "$DATAGEN_ROOT" --work "$WORK" \
  --house-id "$HOUSE_ID" --run-name-prefix "$RUN_PREFIX" --candidate-count "$COUNT" \
  --manifest "${PNP_FIXEDPOOL_MANIFEST:-$WORK/artifacts/seeds/pnp_seed_manifest_fixedpool.json}" \
  --replay-root "${PNP_FIXEDPOOL_REPLAY_ROOT:-$WORK/artifacts/replay_pnp_exact_fixedpool}" \
  --output-hdf5 "${PNP_FIXEDPOOL_OUT:-$WORK/artifacts/seeds/robomimic_pnp_fixedpool_aligned.hdf5}" \
  --action-type "${PNP_SOURCE_ACTION_TYPE:-joint_position}" \
  ${PNP_ALLOW_NONPERSISTENT:+--allow-nonpersistent-candidates}
