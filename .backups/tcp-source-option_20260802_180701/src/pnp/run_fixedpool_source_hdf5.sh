#!/usr/bin/env bash
# Build a MimicGen source HDF5 from an existing fixed-pool Franka datagen collection.
# Raw trajectories are read-only; all generated artifacts are written below the workdir.
set -euo pipefail

ROOT=${MOLMOSPACES_ROOT:?set MOLMOSPACES_ROOT to the repository root}
WORK=${MOLMOSPACES_PNP_WORKDIR:?set MOLMOSPACES_PNP_WORKDIR to a writable output directory}
PY=${MOLMOSPACES_PYTHON:-"$ROOT/.venv/bin/python"}
DATAGEN_ROOT=${1:?usage: $0 /path/to/pick_and_place_planner_v1 [count]}
COUNT=${2:-43}
HOUSE_ID=${PNP_FIXEDPOOL_HOUSE_ID:-1716}
RUN_NAME_PREFIX=${PNP_FIXEDPOOL_RUN_NAME_PREFIX:-"potato_bowl_${HOUSE_ID}_seed"}
MANIFEST=${PNP_FIXEDPOOL_MANIFEST:-"$WORK/artifacts/seeds/pnp_seed_manifest_fixedpool.json"}
REPLAY_ROOT=${PNP_FIXEDPOOL_REPLAY_ROOT:-"$WORK/artifacts/replay_pnp_exact_fixedpool"}
OUT=${PNP_FIXEDPOOL_OUT:-"$WORK/artifacts/seeds/robomimic_pnp_fixedpool_aligned.hdf5"}
LOGDIR="$WORK/logs/fixedpool_source_hdf5_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$(dirname "$MANIFEST")" "$REPLAY_ROOT" "$(dirname "$OUT")" "$LOGDIR"
export PYTHONPATH="$ROOT:${MIMICGEN_ROOT:-$ROOT/vendor/mimicgen}:${ROBOMIMIC_ROOT:-$ROOT/vendor/robomimic}:${PYTHONPATH:-}"

"$PY" "$ROOT/src/pnp/select_pnp_50_source_pool.py" \
  --franka-datagen-root "$DATAGEN_ROOT" --house-id "$HOUSE_ID" --run-name-prefix "$RUN_NAME_PREFIX" --allow-nonpersistent-candidates --count "$COUNT" --out "$MANIFEST" \
  | tee "$LOGDIR/select.log"

failed=0
for ((i = 0; i < COUNT; i++)); do
  seed=$(printf '%02d' "$i")
  "$PY" "$ROOT/src/pnp/collect_homogeneous_datagen_info.py" \
    --seed-index "$i" --manifest "$MANIFEST" --out-root "$REPLAY_ROOT" \
    >"$LOGDIR/seed_${seed}.log" 2>&1 || failed=$((failed + 1))
done

ACCEPTED=$(
  "$PY" - "$MANIFEST" "$REPLAY_ROOT" "$LOGDIR/replay_acceptance.json" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
replay_root = Path(sys.argv[2])
rows = []
accepted = []
for i, seed in enumerate(manifest["seeds"]):
    path = replay_root / f"seed_{i:02d}" / "datagen_info_collection_result.json"
    result = json.loads(path.read_text()) if path.exists() else {"error": "missing replay result"}
    hard_pass = bool(result.get("final_success") and result.get("success_persistent_to_end"))
    if hard_pass:
        accepted.append(i)
    rows.append({"seed_index": i, "source_h5_file": seed.get("source_h5_file"), "traj_key": seed["traj_key"], "hard_pass": hard_pass, "result": result})
Path(sys.argv[3]).write_text(json.dumps({"n_candidates": len(rows), "n_hard_pass": len(accepted), "accepted": accepted, "rows": rows}, indent=2) + "\n")
print(",".join(map(str, accepted)))
PY
)
test -n "$ACCEPTED" || { echo "no replay hard-pass source trajectories" >&2; exit 1; }

"$PY" "$ROOT/src/pnp/convert_seed_set_to_robomimic_50cross.py" \
  --accepted "$ACCEPTED" --manifest "$MANIFEST" --replay-root "$REPLAY_ROOT" --out "$OUT" \
  | tee "$LOGDIR/convert.log"

"$PY" "$ROOT/src/pnp/validate_robomimic_source_hdf5.py" \
  --input "$OUT" --expected-demos "$(tr "," "\n" <<<"$ACCEPTED" | wc -l)" | tee "$LOGDIR/validate.log"

printf 'manifest=%s\nreplay_root=%s\nhdf5=%s\n' "$MANIFEST" "$REPLAY_ROOT" "$OUT" \
  | tee "$LOGDIR/artifacts.txt"
