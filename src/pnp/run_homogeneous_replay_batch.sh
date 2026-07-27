#!/usr/bin/env bash
set -u
cd ${MOLMOSPACES_ROOT}/work/current/mimicgen_pick_and_place || exit 10
LOGDIR="logs/homogeneous_potato_tomato_bowl_replay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "$LOGDIR" > logs/latest_homogeneous_replay_logdir.txt
PY=${MOLMOSPACES_ROOT}/.venv/bin/python
PASS=0
FAIL=0
for i in 0 1 2 3 4 5 6 7 8 9; do
  printf 'START seed_%02d %s\n' "$i" "$(date '+%F %T')" | tee -a "$LOGDIR/batch.log"
  "$PY" src/pnp/collect_homogeneous_datagen_info.py --seed-index "$i" > "$LOGDIR/seed_$(printf '%02d' "$i").log" 2>&1
  code=$?
  echo "$code" > "$LOGDIR/seed_$(printf '%02d' "$i").exit_code"
  if [ "$code" -eq 0 ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
  printf 'DONE seed_%02d code=%s pass=%s fail=%s %s\n' "$i" "$code" "$PASS" "$FAIL" "$(date '+%F %T')" | tee -a "$LOGDIR/batch.log"
  if [ "$code" -ne 0 ]; then
    echo "STOP_ON_FAIL seed_$i" | tee -a "$LOGDIR/batch.log"
    break
  fi
done
printf '{"pass":%s,"fail":%s,"logdir":"%s"}\n' "$PASS" "$FAIL" "$LOGDIR" > "$LOGDIR/summary.json"
cat "$LOGDIR/summary.json"
