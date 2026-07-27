#!/usr/bin/env bash
set -u
ROOT="${MOLMOSPACES_ROOT:-$(pwd)}"
WORK="${MOLMOSPACES_PNP_WORKDIR:-$ROOT/runtime/mimicgen_pick_and_place}"
PY="${MOLMOSPACES_PYTHON:-$ROOT/.venv/bin/python}"
cd "$WORK"
RUN=logs/collect_50cross_datagen_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
echo "$RUN" > logs/latest_collect_50cross_datagen_logdir.txt
: > "$RUN/status.tsv"
for i in $(seq 0 49); do
  out="artifacts/replay_pnp_exact_50cross/seed_$(printf %02d "$i")/datagen_info_collection_result.json"
  if [ -s "$out" ]; then echo -e "$i\tSKIP_EXISTS" | tee -a "$RUN/status.tsv"; continue; fi
  echo -e "$i\tSTART\t$(date -Is)" | tee -a "$RUN/status.tsv"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" "$ROOT/src/pnp/collect_datagen_info_50cross.py" --seed-index "$i" > "$RUN/seed_$(printf %02d "$i").log" 2>&1
  code=$?
  echo "$code" > "$RUN/seed_$(printf %02d "$i").exit_code"
  echo -e "$i\tEXIT_$code\t$(date -Is)" | tee -a "$RUN/status.tsv"
  if [ "$code" -ne 0 ]; then echo "STOP_ON_FAIL seed=$i" | tee -a "$RUN/status.tsv"; exit "$code"; fi
done
