#!/usr/bin/env bash
set -euo pipefail

: "${MOLMOSPACES_ROOT:?Set MOLMOSPACES_ROOT to the MolmoSpaces checkout}"
WORK="${MOLMOSPACES_PNP_WORKDIR:-${MOLMOSPACES_ROOT}/work/current/mimicgen_pick_and_place}"
PY="${MOLMOSPACES_PYTHON:-${MOLMOSPACES_ROOT}/.venv/bin/python}"
cd "$WORK"

SRC="${PNP_SOURCE_HDF5:-artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5}"
MAN="${PNP_TARGET_MANIFEST:-artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json}"
OLD_RUN="${PREVIOUS_COLLECTOR_RUN:?Set PREVIOUS_COLLECTOR_RUN to a collector log directory containing accepted.jsonl}"
TARGET_SUCCESS="${TARGET_SUCCESS:-100}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-500}"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN="logs/collect_unique_highyield_successes_${STAMP}"
ART_PREFIX="collect_unique_highyield_successes_${STAMP}"
mkdir -p "$RUN"
echo "$RUN" > logs/latest_collect_unique_highyield_successes_logdir.txt
ATT="$RUN/attempts.jsonl"
ACC="$RUN/accepted.jsonl"
HASHES="$RUN/action_hashes.txt"
: > "$ATT"
: > "$ACC"
: > "$HASHES"

python3 - "$OLD_RUN/accepted.jsonl" "$ACC" "$HASHES" <<'PY'
import hashlib, json, pathlib, sys
old_acc = pathlib.Path(sys.argv[1])
acc_path = pathlib.Path(sys.argv[2])
hash_path = pathlib.Path(sys.argv[3])
seen = set()
accepted = []
for line in old_acc.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    artifact = pathlib.Path(row.get('artifact_dir', ''))
    if not artifact.is_absolute():
        artifact = pathlib.Path.cwd() / artifact
    action_file = artifact / 'generated_actions.npy'
    if not action_file.exists():
        continue
    action_hash = hashlib.sha256(action_file.read_bytes()).hexdigest()
    if action_hash in seen:
        continue
    seen.add(action_hash)
    row = dict(row)
    row['accepted_index'] = None
    row['source'] = 'seeded_unique_from_previous_collectors'
    row['action_sha256'] = action_hash
    accepted.append(row)
with acc_path.open('w') as out:
    for i, row in enumerate(accepted, 1):
        row['accepted_index'] = i
        out.write(json.dumps(row, ensure_ascii=False) + '\n')
hash_path.write_text('\n'.join(sorted(seen)) + ('\n' if seen else ''))
print(len(accepted))
PY

renumber_accepts() {
  python3 - "$ACC" "$RUN/accepted.tmp" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
with open(sys.argv[2], 'w') as f:
    for i, row in enumerate(rows, 1):
        row['accepted_index'] = i
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
  mv "$RUN/accepted.tmp" "$ACC"
}
accepted_count() { python3 - "$ACC" <<'PY'
import sys
print(sum(1 for l in open(sys.argv[1]) if l.strip()))
PY
}
write_summary() {
  python3 - "$ATT" "$ACC" "$RUN/summary.json" "$TARGET_SUCCESS" <<'PY'
import json, pathlib, sys
attempts = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
accepted = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
unique = sum(1 for row in attempts if row.get('accepted_unique'))
summary = {
    'target_success': int(sys.argv[4]),
    'accepted_total_unique': len(accepted),
    'accepted_seeded_unique': sum(1 for row in accepted if row.get('source') == 'seeded_unique_from_previous_collectors'),
    'accepted_new_unique': sum(1 for row in accepted if row.get('source') == 'collect_unique_highyield_successes'),
    'new_attempts': len(attempts),
    'new_success': sum(1 for row in attempts if row.get('revised_success')),
    'new_duplicate_success': sum(1 for row in attempts if row.get('duplicate_action')),
    'new_unique_success': unique,
    'new_unique_success_rate': unique / len(attempts) if attempts else None,
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

CELLS=(
  "2|demo_2|target02_src02"
  "2|demo_1|target02_src01"
  "1|demo_1|target01_src01"
  "3|demo_3|target03_src03"
  "4|demo_4|target04_src04"
  "8|demo_9|target08_src09"
  "2|demo_0|target02_src00"
  "3|demo_0|target03_src00"
  "0|demo_0|target00_src00"
  "0|demo_1|target00_src01"
  "8|demo_8|target08_src08"
  "9|demo_9|target09_src09"
  "5|demo_5|target05_src05"
  "6|demo_6|target06_src06"
  "7|demo_7|target07_src07"
)
INTERPS=(0 1 2 3 4 6 8)
FIXEDS=(0 1 2 4)
CURRENT_FLAGS=(0 1)

renumber_accepts
attempt=0
echo "START run=$RUN seeded_unique=$(accepted_count) target_success=$TARGET_SUCCESS" | tee -a "$RUN/collector.log"
for current_flag in "${CURRENT_FLAGS[@]}"; do
  for interp in "${INTERPS[@]}"; do
    for fixed in "${FIXEDS[@]}"; do
      for cell in "${CELLS[@]}"; do
        [ "$(accepted_count)" -lt "$TARGET_SUCCESS" ] || break 4
        [ "$attempt" -lt "$MAX_ATTEMPTS" ] || break 4
        IFS='|' read -r target demo_keys label <<< "$cell"
        attempt=$((attempt + 1))
        out_name=$(printf "%s_attempt%04d_%s_i%02d_f%02d_c%d" "$ART_PREFIX" "$attempt" "$label" "$interp" "$fixed" "$current_flag")
        extra=()
        if [ "$current_flag" = "1" ]; then extra+=(--interpolate-from-current-pose); fi
        echo "ATTEMPT start attempt=$attempt accepted=$(accepted_count) target=$target demo_keys=$demo_keys interp=$interp fixed=$fixed current=$current_flag out=$out_name" | tee -a "$RUN/collector.log"
        set +e
        "$PY" src/pnp/generate_pick_place_rollout.py \
          --source-hdf5 "$SRC" \
          --target-manifest "$MAN" \
          --demo-keys "$demo_keys" \
          --seed-index "$target" \
          --out-name "$out_name" \
          --interp "$interp" --fixed "$fixed" --noise 0.0 \
          --transform-first-robot-pose \
          --post-hold-steps 30 \
          --save-videos "${extra[@]}" > "$RUN/$out_name.log" 2>&1
        code=$?
        set -e
        python3 - "$out_name" "$code" "$target" "$demo_keys" "$interp" "$fixed" "$current_flag" "$HASHES" >> "$ATT" <<'PY'
import hashlib, json, pathlib, sys
name = sys.argv[1]
code = int(sys.argv[2])
target = int(sys.argv[3])
demo_keys = sys.argv[4]
interp = int(sys.argv[5])
fixed = int(sys.argv[6])
current = bool(int(sys.argv[7]))
hash_file = pathlib.Path(sys.argv[8])
seen = set(hash_file.read_text().splitlines()) if hash_file.exists() else set()
artifact = pathlib.Path.cwd() / 'artifacts/mimicgen_pnp' / name
result = artifact / 'generate_result.json'
trace_file = artifact / 'success_trace.json'
action_file = artifact / 'generated_actions.npy'
row = {'name': name, 'code': code, 'missing_result': not result.exists(), 'requested_target': target, 'requested_demo_keys': demo_keys, 'interp': interp, 'fixed': fixed, 'interpolate_from_current_pose': current}
if result.exists():
    result_data = json.load(open(result))
    trace = json.load(open(trace_file)) if trace_file.exists() else []
    tail30_success = bool(trace) and all(x.get('success') for x in trace[-30:])
    action_hash = hashlib.sha256(action_file.read_bytes()).hexdigest() if action_file.exists() else None
    revised = bool(result_data.get('final_success') and tail30_success)
    duplicate = bool(revised and action_hash in seen)
    row.update({
        'target': result_data.get('generation_env_seed_index'),
        'house': result_data.get('generation_env_house_id'),
        'src_demo_inds': result_data.get('src_demo_inds'),
        'src': result_data.get('source_demo_keys'),
        'final': bool(result_data.get('final_success')),
        'strict_persist': bool(result_data.get('success_persistent_to_end')),
        'first': result_data.get('first_success_step'),
        'actions': result_data.get('num_actions_executed'),
        'tail30_success': tail30_success,
        'revised_success': revised,
        'action_sha256': action_hash,
        'duplicate_action': duplicate,
        'accepted_unique': bool(revised and action_hash and not duplicate),
    })
print(json.dumps(row, ensure_ascii=False))
PY
        status=$(python3 - "$ATT" <<'PY'
import json, sys
last = None
for line in open(sys.argv[1]):
    if line.strip(): last = json.loads(line)
if not last or not last.get('revised_success'):
    print('reject')
elif last.get('duplicate_action'):
    print('duplicate')
elif last.get('accepted_unique'):
    print('accept')
else:
    print('reject')
PY
)
        if [ "$status" = "accept" ]; then
          python3 - "$ATT" "$ACC" "$HASHES" "$out_name" <<'PY'
import json, pathlib, sys
last = None
for line in open(sys.argv[1]):
    if line.strip(): last = json.loads(line)
row = {
    'accepted_index': None,
    'source': 'collect_unique_highyield_successes',
    'name': sys.argv[4],
    'artifact_dir': str(pathlib.Path.cwd() / 'artifacts/mimicgen_pnp' / sys.argv[4]),
    **{key: last.get(key) for key in ['requested_target','requested_demo_keys','interp','fixed','interpolate_from_current_pose','target','house','src','src_demo_inds','final','strict_persist','tail30_success','revised_success','first','actions','action_sha256']},
}
with open(sys.argv[2], 'a') as f:
    f.write(json.dumps(row, ensure_ascii=False) + '\n')
with open(sys.argv[3], 'a') as f:
    f.write(str(last.get('action_sha256')) + '\n')
PY
          renumber_accepts
          echo "ATTEMPT accepted_unique attempt=$attempt accepted=$(accepted_count) out=$out_name" | tee -a "$RUN/collector.log"
        else
          rm -rf "artifacts/mimicgen_pnp/$out_name"
          echo "ATTEMPT $status attempt=$attempt accepted=$(accepted_count) out=$out_name code=$code" | tee -a "$RUN/collector.log"
        fi
        write_summary
      done
    done
  done
done
write_summary
echo "FINISH accepted=$(accepted_count) attempts=$attempt" | tee -a "$RUN/collector.log"
