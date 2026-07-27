#!/usr/bin/env bash
set -u
ROOT="${MOLMOSPACES_ROOT:-$(pwd)}"
WORK="${MOLMOSPACES_PNP_WORKDIR:-$ROOT/runtime/mimicgen_pick_and_place}"
PY="${MOLMOSPACES_PYTHON:-$ROOT/.venv/bin/python}"
cd "$WORK"
RUN=logs/collect_50cross_datagen_parallel_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
echo "$RUN" > logs/latest_collect_50cross_datagen_logdir.txt
echo "run_dir=$RUN" > "$RUN/summary.txt"
echo "parallel_workers=4" >> "$RUN/summary.txt"
echo "cuda_visible_devices=0" >> "$RUN/summary.txt"
seq 0 49 > "$RUN/queue.txt"
worker() {
  wid="$1"
  while true; do
    {
      flock -x 9
      seed=""
      while read -r cand; do
        [ -z "$cand" ] && continue
        flag="$RUN/claimed_${cand}"
        doneflag="$RUN/done_${cand}"
        if [ -e "$flag" ] || [ -e "$doneflag" ]; then continue; fi
        seed="$cand"; echo "$wid" > "$flag"; break
      done < "$RUN/queue.txt"
      echo "$seed"
    } 9>"$RUN/claim.lock" > "$RUN/worker_${wid}.next"
    seed=$(cat "$RUN/worker_${wid}.next")
    [ -z "$seed" ] && break
    ss=$(printf "%02d" "$seed")
    out="artifacts/replay_pnp_exact_50cross/seed_${ss}/datagen_info_collection_result.json"
    if [ -s "$out" ]; then
      echo -e "$(date -Is)\tworker=$wid\tseed=$seed\tSKIP_EXISTS" >> "$RUN/status.tsv"
      touch "$RUN/done_${seed}"
      continue
    fi
    echo -e "$(date -Is)\tworker=$wid\tseed=$seed\tSTART" >> "$RUN/status.tsv"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" "$ROOT/src/pnp/collect_datagen_info_50cross.py" --seed-index "$seed" > "$RUN/seed_${ss}.log" 2>&1
    code=$?
    echo "$code" > "$RUN/seed_${ss}.exit_code"
    echo -e "$(date -Is)\tworker=$wid\tseed=$seed\tEXIT_$code" >> "$RUN/status.tsv"
    touch "$RUN/done_${seed}"
  done
  echo -e "$(date -Is)\tworker=$wid\tDONE" >> "$RUN/status.tsv"
}
for wid in 0 1 2 3; do worker "$wid" & echo $! >> "$RUN/worker_pids.txt"; done
wait
failed=$(find "$RUN" -maxdepth 1 -name "seed_*.exit_code" -exec sh -c "for f; do [ \"$(cat \"$f\")\" = 0 ] || echo \"$f:$(cat \"$f\")\"; done" sh {} +)
if [ -n "$failed" ]; then echo "$failed" > "$RUN/failures.txt"; exit 2; fi
echo "ALL_DONE" > "$RUN/complete.txt"
