#!/usr/bin/env bash
set -euo pipefail

# MimicGen per-subtask-selection expansion from the approved 43-demo source HDF5.
# Source provenance remains recorded per subtask. Acceptance follows the standard
# MimicGen execution path plus full simulator/video/post-hold and action-hash gates.

: "${MOLMOSPACES_ROOT:?Set MOLMOSPACES_ROOT to the canonical integration checkout}"
ROOT="$MOLMOSPACES_ROOT"
WORK="${MOLMOSPACES_PNP_WORKDIR:-$ROOT/runtime/fixedpool_potato_bowl_1716_source}"
PY="${MOLMOSPACES_PYTHON:-$ROOT/.venv/bin/python}"
SOURCE_HDF5="${PNP_SOURCE_HDF5:-$WORK/artifacts/seeds/robomimic_pnp_fixedpool_manual_review43.hdf5}"
TARGET_MANIFEST="${PNP_TARGET_MANIFEST:-$WORK/artifacts/seeds/pnp_seed_manifest_fixedpool.json}"
TARGET_SUCCESS="${TARGET_SUCCESS:-100}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-240}"
RUN_DIR="${RUN_DIR:-$WORK/logs/fixedpool_mimicgen43_to100_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$WORK/logs/latest_fixedpool_mimicgen43_to100_logdir.txt"
ATTEMPTS="$RUN_DIR/attempts.jsonl"
ACCEPTED="$RUN_DIR/accepted.jsonl"
HASHES="$RUN_DIR/action_hashes.txt"
touch "$ATTEMPTS" "$ACCEPTED" "$HASHES"
SOURCE_SHA256=$(sha256sum "$SOURCE_HDF5" | awk '{print $1}')
DEMO_KEYS=$("$PY" - <<'PY'
print(",".join(f"demo_{i}" for i in range(43)))
PY
)

accepted_count() {
  "$PY" - "$ACCEPTED" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1]) if line.strip()))
PY
}

attempted_name() {
  "$PY" - "$ATTEMPTS" "$1" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    if line.strip() and json.loads(line).get("name") == sys.argv[2]:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

write_summary() {
  "$PY" - "$ATTEMPTS" "$ACCEPTED" "$RUN_DIR/summary.json" "$TARGET_SUCCESS" "$SOURCE_HDF5" "$SOURCE_SHA256" <<'PY'
import json, pathlib, sys
attempts = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
accepted = [json.loads(x) for x in open(sys.argv[2]) if x.strip()]
summary = {
    "source_hdf5": sys.argv[5], "source_hdf5_sha256": sys.argv[6],
    "target_success": int(sys.argv[4]), "attempts": len(attempts),
    "strict_successes": sum(bool(x.get("strict_success")) for x in attempts),
    "single_source_selection_successes": sum(bool(x.get("single_source_selection_success")) for x in attempts),
    "duplicate_successes": sum(bool(x.get("duplicate_action")) for x in attempts),
    "accepted_unique_cross_subtask": len(accepted),
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False))
PY
}

record_attempt() {
  local name="$1" target="$2" rng_seed="$3" code="$4"
  "$PY" - "$name" "$target" "$rng_seed" "$code" "$WORK" "$HASHES" "$SOURCE_SHA256" <<'PY' >> "$ATTEMPTS"
import hashlib, json, pathlib, sys
name, target, rng_seed, code, work, hash_file, source_sha = sys.argv[1:]
artifact = pathlib.Path(work) / "artifacts/mimicgen_pnp" / name
result_file, trace_file, action_file = (artifact / x for x in ("generate_result.json", "success_trace.json", "generated_actions.npy"))
videos = sorted(artifact.glob("*.mp4"))
seen = {x.strip() for x in pathlib.Path(hash_file).read_text().splitlines() if x.strip()}
row = {"name": name, "requested_target": int(target), "mimicgen_rng_seed": int(rng_seed), "exit_code": int(code), "artifact_dir": str(artifact), "source_hdf5_sha256": source_sha, "has_result": result_file.exists()}
if result_file.exists():
    result = json.loads(result_file.read_text())
    trace = json.loads(trace_file.read_text()) if trace_file.exists() else []
    src_demo_inds = [int(x) for x in result.get("src_demo_inds", [])]
    distinct_sources = len(set(src_demo_inds))
    tail30 = len(trace) >= 30 and all(bool(x.get("success")) for x in trace[-30:])
    action_sha = hashlib.sha256(action_file.read_bytes()).hexdigest() if action_file.exists() else None
    videos_ok = len(videos) >= 2 and all(p.stat().st_size > 0 for p in videos)
    base_gate = bool(result.get("select_src_per_subtask") and result.get("final_success") and result.get("success_persistent_to_end") and tail30 and videos_ok and action_sha)
    strict = base_gate
    duplicate = bool(strict and action_sha in seen)
    row.update({
        "target_seed_index": result.get("generation_env_seed_index"), "house_id": result.get("generation_env_house_id"),
        "src_demo_inds": src_demo_inds, "distinct_src_demo_count": distinct_sources,
        "final_success": bool(result.get("final_success")), "success_persistent_to_end": bool(result.get("success_persistent_to_end")),
        "tail30_success": tail30, "videos_ok": videos_ok, "num_actions": result.get("num_actions_executed"),
        "action_sha256": action_sha,
        "single_source_selection_success": bool(base_gate and distinct_sources == 1),
        "strict_success": strict, "duplicate_action": duplicate, "accepted_unique": bool(strict and not duplicate),
    })
print(json.dumps(row, ensure_ascii=False))
PY
}

append_accept_if_eligible() {
  "$PY" - "$ATTEMPTS" "$ACCEPTED" "$HASHES" <<'PY'
import json, pathlib, sys
attempts, accepted, hashes = map(pathlib.Path, sys.argv[1:])
row = next((json.loads(x) for x in reversed(attempts.read_text().splitlines()) if x.strip()), None)
if not row or not row.get("accepted_unique"):
    raise SystemExit(1)
row = dict(row)
row["accepted_index"] = sum(1 for x in accepted.read_text().splitlines() if x.strip()) + 1
with accepted.open("a") as out: out.write(json.dumps(row, ensure_ascii=False) + "\n")
with hashes.open("a") as out: out.write(row["action_sha256"] + "\n")
PY
}

run_one() {
  local target="$1" rng_seed="$2" name
  name=$(printf 'fixedpool43_crosssubtask_target%02d_rng%05d' "$target" "$rng_seed")
  attempted_name "$name" && return 0
  echo "$(date -Is) START name=$name target=$target rng_seed=$rng_seed accepted=$(accepted_count)" | tee -a "$RUN_DIR/collector.log"
  set +e
  "$PY" src/pnp/generate_pick_place_rollout.py \
    --seed-index "$target" --out-name "$name" \
    --source-hdf5 "$SOURCE_HDF5" --target-manifest "$TARGET_MANIFEST" \
    --demo-keys "$DEMO_KEYS" --select-src-per-subtask --mimicgen-rng-seed "$rng_seed" \
    --transform-first-robot-pose --post-hold-steps 30 --save-videos > "$RUN_DIR/$name.log" 2>&1
  code=$?
  set -e
  record_attempt "$name" "$target" "$rng_seed" "$code"
  if append_accept_if_eligible; then
    echo "$(date -Is) ACCEPT name=$name accepted=$(accepted_count)" | tee -a "$RUN_DIR/collector.log"
  else
    echo "$(date -Is) REJECT name=$name code=$code" | tee -a "$RUN_DIR/collector.log"
  fi
  write_summary >> "$RUN_DIR/collector.log"
}

echo "START run=$RUN_DIR target_success=$TARGET_SUCCESS max_attempts=$MAX_ATTEMPTS source_sha256=$SOURCE_SHA256 mode=official_per_subtask_selection" | tee -a "$RUN_DIR/collector.log"
for attempt in $(seq 0 $((MAX_ATTEMPTS - 1))); do
  [ "$(accepted_count)" -lt "$TARGET_SUCCESS" ] || break
  target=$((attempt % 43))
  rng_seed=$((10000 + attempt))
  run_one "$target" "$rng_seed"
done
write_summary | tee -a "$RUN_DIR/collector.log"
echo "FINISH accepted=$(accepted_count)" | tee -a "$RUN_DIR/collector.log"
