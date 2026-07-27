#!/usr/bin/env bash
set -u
ROOT="${MOLMOSPACES_ROOT:-$(pwd)}"
WORK="${MOLMOSPACES_PNP_WORKDIR:-$ROOT/runtime/mimicgen_pick_and_place}"
PY="${MOLMOSPACES_PYTHON:-$ROOT/.venv/bin/python}"
cd "$WORK"
RUN=$(cat logs/latest_collect_50cross_datagen_logdir.txt)
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
        seed="$cand"
        echo "$wid" > "$flag"
        break
      done < "$RUN/queue.txt"
      echo "$seed"
    } 9>"$RUN/claim.lock" > "$RUN/extra_worker_${wid}.next"
    seed=$(cat "$RUN/extra_worker_${wid}.next")
    [ -z "$seed" ] && break
    ss=$(printf "%02d" "$seed")
    out="artifacts/replay_pnp_exact_50cross/seed_${ss}/datagen_info_collection_result.json"
    if [ -s "$out" ]; then
      echo -e "$(date -Is)\textra_worker=$wid\tseed=$seed\tSKIP_EXISTS" >> "$RUN/status.tsv"
      touch "$RUN/done_${seed}"
      continue
    fi
    echo -e "$(date -Is)\textra_worker=$wid\tseed=$seed\tSTART" >> "$RUN/status.tsv"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" "$ROOT/src/pnp/collect_datagen_info_50cross.py" --seed-index "$seed" > "$RUN/seed_${ss}.log" 2>&1
    code=$?
    echo "$code" > "$RUN/seed_${ss}.exit_code"
    echo -e "$(date -Is)\textra_worker=$wid\tseed=$seed\tEXIT_$code" >> "$RUN/status.tsv"
    touch "$RUN/done_${seed}"
  done
  echo -e "$(date -Is)\textra_worker=$wid\tDONE" >> "$RUN/status.tsv"
}
for wid in 4 5 6 7; do
  worker "$wid" & echo $! >> "$RUN/extra_worker_pids.txt"
done
wait
