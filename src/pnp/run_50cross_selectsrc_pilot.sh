#!/usr/bin/env bash
set -u
ROOT="${MOLMOSPACES_ROOT:-$(pwd)}"
WORK="${MOLMOSPACES_PNP_WORKDIR:-$ROOT/runtime/mimicgen_pick_and_place}"
PY="${MOLMOSPACES_PYTHON:-$ROOT/.venv/bin/python}"
cd "$WORK"
RUN=logs/50cross_selectsrc_pilot_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
echo "$RUN" > logs/latest_50cross_selectsrc_pilot_logdir.txt
SRC=artifacts/seeds/robomimic_pnp_50demo_crossmix_aligned.hdf5
MAN=artifacts/seeds/pnp_seed_manifest_50demo_crossmix.json
DEMO_KEYS=$("$PY" - <<"PY"
print(",".join(f"demo_{i}" for i in range(50)))
PY
)
echo "run_dir=$RUN" | tee -a "$RUN/summary.tsv"
echo "source_hdf5=$SRC" | tee -a "$RUN/summary.tsv"
echo "target_manifest=$MAN" | tee -a "$RUN/summary.tsv"
for seed in 0 1 2 3 4 5 6 7 8 9; do
  ss=$(printf "%02d" "$seed")
  out_name="gen_50cross_selectsrc_seed${ss}"
  echo -e "$(date -Is)\tseed=$seed\tSTART" | tee -a "$RUN/status.tsv"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" "$ROOT/src/pnp/generate_pick_place_rollout.py" \
    --seed-index "$seed" \
    --out-name "$out_name" \
    --source-hdf5 "$SRC" \
    --target-manifest "$MAN" \
    --demo-keys "$DEMO_KEYS" \
    --select-src-per-subtask \
    --transform-first-robot-pose \
    --post-hold-steps 30 \
    --save-videos \
    > "$RUN/seed_${ss}.log" 2>&1
  code=$?
  echo "$code" > "$RUN/seed_${ss}.exit_code"
  final=false; persist=false
  if [ -s "artifacts/mimicgen_pnp/$out_name/result.json" ]; then
    final=$($PY - <<PY
import json
p="artifacts/mimicgen_pnp/$out_name/result.json"
d=json.load(open(p))
print(bool(d.get("final_success")))
PY
)
    persist=$($PY - <<PY
import json
p="artifacts/mimicgen_pnp/$out_name/result.json"
d=json.load(open(p))
print(bool(d.get("success_persistent_to_end")))
PY
)
  fi
  echo -e "$(date -Is)\tseed=$seed\tEXIT_$code\tfinal=$final\tpersist=$persist\tout=$out_name" | tee -a "$RUN/status.tsv"
done
"$PY" - <<"PY"
import json, pathlib
run=pathlib.Path("logs/latest_50cross_selectsrc_pilot_logdir.txt").read_text().strip()
rows=[]
for p in sorted(pathlib.Path(run).glob("seed_*.exit_code")):
    seed=int(p.stem.split("_")[1])
    out=pathlib.Path("artifacts/mimicgen_pnp")/f"gen_50cross_selectsrc_seed{seed:02d}"/"result.json"
    row={"seed":seed,"code":int(p.read_text().strip() or 999),"has_result":out.exists()}
    if out.exists():
        d=json.loads(out.read_text())
        row.update({k:d.get(k) for k in ["final_success","success_persistent_to_end","num_actions","select_src_per_subtask","transform_first_robot_pose","post_hold_steps"]})
    rows.append(row)
summary={"run":run,"rows":rows,"strict_success":sum(1 for r in rows if r.get("code")==0 and r.get("final_success") and r.get("success_persistent_to_end")),"n":len(rows)}
path=pathlib.Path(run)/"summary.json"
path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
