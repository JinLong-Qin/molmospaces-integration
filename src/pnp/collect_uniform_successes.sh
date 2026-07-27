#!/usr/bin/env bash
set -euo pipefail

: "${MOLMOSPACES_ROOT:?Set MOLMOSPACES_ROOT to the MolmoSpaces checkout}"
WORK="${MOLMOSPACES_PNP_WORKDIR:-${MOLMOSPACES_ROOT}/work/current/mimicgen_pick_and_place}"
PY="${MOLMOSPACES_PYTHON:-${MOLMOSPACES_ROOT}/.venv/bin/python}"
cd "$WORK"

SRC="${PNP_SOURCE_HDF5:-artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5}"
MAN="${PNP_TARGET_MANIFEST:-artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json}"
PILOT_LOGDIR="${PNP_PILOT_LOGDIR:-logs/foodlike_bowl_100pilot_20260727_015725}"
TARGET_SUCCESS="${TARGET_SUCCESS:-100}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-800}"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN="logs/collect_uniform_successes_${STAMP}"
ART_PREFIX="collect_uniform_successes_${STAMP}"
mkdir -p "$RUN"
echo "$RUN" > logs/latest_collect_uniform_successes_logdir.txt
ATT="$RUN/attempts.jsonl"
ACC="$RUN/accepted.jsonl"
: > "$ATT"
: > "$ACC"

python3 - "$ACC" "$PILOT_LOGDIR" <<'PY'
import json, pathlib, sys
acc_path = pathlib.Path(sys.argv[1])
pilot = pathlib.Path(sys.argv[2])
base = pathlib.Path.cwd()
rows = [json.loads(l) for l in open(pilot / 'results.jsonl') if l.strip()]
with acc_path.open('a') as out:
    for row in rows:
        artifact = base / 'artifacts/mimicgen_pnp' / row['name']
        trace = json.load(open(artifact / 'success_trace.json'))
        tail_ok = bool(trace) and all(x.get('success') for x in trace[-30:])
        if row.get('final') and tail_ok:
            out.write(json.dumps({
                'accepted_index': None,
                'source': pilot.name,
                'name': row['name'],
                'artifact_dir': str(artifact),
                'target': row.get('target'),
                'house': row.get('house'),
                'src': row.get('src'),
                'final': row.get('final'),
                'strict_persist': row.get('persist'),
                'tail30_success': tail_ok,
                'revised_success': True,
            }, ensure_ascii=False) + '\n')
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
target_success = int(sys.argv[4])
new_success = sum(1 for row in attempts if row.get('revised_success'))
summary = {
    'target_success': target_success,
    'accepted_total': len(accepted),
    'accepted_seeded_from_pilot': sum(1 for row in accepted if row.get('source') != 'collect_uniform_successes'),
    'accepted_new': sum(1 for row in accepted if row.get('source') == 'collect_uniform_successes'),
    'new_attempts': len(attempts),
    'new_revised_success': new_success,
    'new_success_rate': new_success / len(attempts) if attempts else None,
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

renumber_accepts
attempt=0
echo "START run=$RUN initial_accepted=$(accepted_count) target_success=$TARGET_SUCCESS" | tee -a "$RUN/collector.log"
while [ "$(accepted_count)" -lt "$TARGET_SUCCESS" ] && [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt + 1))
  target=$(( (attempt - 1) % 10 ))
  out_name=$(printf "%s_attempt%04d_target%02d" "$ART_PREFIX" "$attempt" "$target")
  echo "ATTEMPT start attempt=$attempt accepted=$(accepted_count) target=$target out=$out_name" | tee -a "$RUN/collector.log"
  set +e
  "$PY" src/pnp/generate_pick_place_rollout.py \
    --source-hdf5 "$SRC" \
    --target-manifest "$MAN" \
    --demo-keys "demo_0,demo_1,demo_2,demo_3,demo_4,demo_5,demo_6,demo_7,demo_8,demo_9" \
    --seed-index "$target" \
    --out-name "$out_name" \
    --interp 0 --fixed 0 --noise 0.0 \
    --transform-first-robot-pose \
    --post-hold-steps 30 \
    --save-videos > "$RUN/$out_name.log" 2>&1
  code=$?
  set -e
  python3 - "$out_name" "$code" >> "$ATT" <<'PY'
import json, pathlib, sys
name = sys.argv[1]
code = int(sys.argv[2])
base = pathlib.Path.cwd() / 'artifacts/mimicgen_pnp' / name
result = base / 'generate_result.json'
trace_file = base / 'success_trace.json'
row = {'name': name, 'code': code, 'missing_result': not result.exists()}
if result.exists():
    result_data = json.load(open(result))
    trace = json.load(open(trace_file)) if trace_file.exists() else []
    tail30_success = bool(trace) and all(x.get('success') for x in trace[-30:])
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
        'revised_success': bool(result_data.get('final_success') and tail30_success),
    })
print(json.dumps(row, ensure_ascii=False))
PY
  revised=$(python3 - "$ATT" <<'PY'
import json, sys
last = None
for line in open(sys.argv[1]):
    if line.strip():
        last = json.loads(line)
print('1' if last and last.get('revised_success') else '0')
PY
)
  if [ "$revised" = "1" ]; then
    python3 - "$ATT" "$ACC" "$out_name" <<'PY'
import json, pathlib, sys
last = None
for line in open(sys.argv[1]):
    if line.strip(): last = json.loads(line)
row = {
    'accepted_index': None,
    'source': 'collect_uniform_successes',
    'name': sys.argv[3],
    'artifact_dir': str(pathlib.Path.cwd() / 'artifacts/mimicgen_pnp' / sys.argv[3]),
    **{key: last.get(key) for key in ['target','house','src','src_demo_inds','final','strict_persist','tail30_success','revised_success','first','actions']},
}
with open(sys.argv[2], 'a') as f:
    f.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
    renumber_accepts
    echo "ATTEMPT accepted attempt=$attempt accepted=$(accepted_count) out=$out_name" | tee -a "$RUN/collector.log"
  else
    rm -rf "artifacts/mimicgen_pnp/$out_name"
    echo "ATTEMPT rejected attempt=$attempt accepted=$(accepted_count) out=$out_name code=$code" | tee -a "$RUN/collector.log"
  fi
  write_summary
done
write_summary
echo "FINISH accepted=$(accepted_count) attempts=$attempt" | tee -a "$RUN/collector.log"
