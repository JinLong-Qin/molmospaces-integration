# Pick-and-Place MimicGen Scripts

This directory is the code entrypoint for the MimicGen Pick-and-Place worklines.

Canonical workline READMEs:

- [`docs/worklines/mimicgen_pick_and_place/README.md`](../../docs/worklines/mimicgen_pick_and_place/README.md)
- [`docs/worklines/mimicgen_50cross/README.md`](../../docs/worklines/mimicgen_50cross/README.md)

## Standard environment variables

Run from the repository root after `bash tools/setup_mimicgen_dependency.sh`:

```bash
export MOLMOSPACES_ROOT="$PWD"
export MOLMOSPACES_PNP_WORKDIR="$PWD/runtime/mimicgen_pick_and_place"
export MIMICGEN_ROOT="$PWD/vendor/mimicgen"
export ROBOMIMIC_ROOT="$PWD/vendor/robomimic"
export PYTHONPATH="$PWD:$MIMICGEN_ROOT:$ROBOMIMIC_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
export MOLMOSPACES_NLTK_DATA="$NLTK_DATA"
```

## Execution architecture

New experiments must use the two parameterized Python orchestrators:

- `run_generation.py` is the single rollout runner. It owns target iteration, source selection, result inspection, action/layout deduplication, JSONL provenance, and summary generation. Use `--mode per-subtask` for official MimicGen source selection or `--mode whole-source --diagnostic` for the quarantined whole-source control.
- `run_source_hdf5_pipeline.py` is the single source-build orchestrator. It parameterizes candidate selection, deterministic replay, acceptance recording, robomimic conversion, and validation.

The remaining `run_*.sh` generation files are compatibility wrappers with experiment-specific defaults only. They must not contain Python heredocs or new acceptance logic. Runtime artifacts remain under the selected `runtime/` work root; source pools, target manifests, and diagnostic runs must use separate paths and labels.

Example parameterized generation commands:

```bash
python src/pnp/run_generation.py --root "$MOLMOSPACES_ROOT" \
  --python "$MOLMOSPACES_PYTHON" --work "$MOLMOSPACES_PNP_WORKDIR" \
  --mode per-subtask --source-hdf5 /path/source.hdf5 \
  --target-manifest /path/targets.json --target-success 10 \
  --target-start 0 --target-end 9 --run-label foodlike_bowl_pilot

python src/pnp/run_generation.py --root "$MOLMOSPACES_ROOT" \
  --python "$MOLMOSPACES_PYTHON" --work "$MOLMOSPACES_PNP_WORKDIR" \
  --mode whole-source --diagnostic --source-count 17 \
  --source-hdf5 /path/source.hdf5 --target-manifest /path/targets.json \
  --target-success 10 --run-label fixedbase_wholesource_control
```

### Source inspection and replay

- `inspect_source_candidates.py`
- `select_pick_place_seeds.py`
- `replay_source_episode.py`
- `run_homogeneous_replay_batch.sh`
- `run_homogeneous_replay_remaining.sh`

### MimicGen / robomimic conversion

- `collect_datagen_info.py`
- `collect_homogeneous_datagen_info.py`
- `convert_seed_set_to_robomimic.py`
- `convert_single_seed_to_robomimic.py`
- `parse_source_dataset.py`

### Generated rollout execution

- `generate_pick_place_rollout.py` - one simulator rollout; this is the execution primitive, not a batch orchestrator.
- `run_generation.py` - canonical parameterized batch/pilot orchestrator.
- `collect_uniform_successes.sh` and `collect_unique_highyield_successes.sh` - legacy collectors retained for historical provenance; do not use them as templates for new work.

### 50-demo cross-subtask diagnostics

- `select_pnp_50_source_pool.py`
- `collect_datagen_info_50cross.py`
- `convert_seed_set_to_robomimic_50cross.py`
- `run_collect_50cross_datagen.sh`
- `run_collect_50cross_datagen_parallel.sh`
- `run_collect_50cross_extra_workers.sh`
- `run_50cross_selectsrc_pilot.sh`

### Fixed-pool source-HDF5 build

- `run_source_hdf5_pipeline.py` - canonical parameterized selector -> replay -> conversion -> validation orchestrator.
- `run_fixedpool_source_hdf5.sh` - thin compatibility wrapper around the canonical source pipeline.
- `validate_robomimic_source_hdf5.py` - verifies the generated `data/demo_*` HDF5 structure, alignment, finite numeric arrays, and source provenance.

### Historical/legacy diagnostics

The 50-cross pilot launchers and old collector scripts remain only to reproduce dated evidence. Their outputs are diagnostic unless the corresponding project gate explicitly accepts them. New work should express differences through CLI arguments such as source HDF5, target manifest, target range, source count, RNG base, run label, and diagnostic mode.

## Quick checks

```bash
python src/pnp/parse_source_dataset.py --help
python src/pnp/generate_pick_place_rollout.py --help
bash -n src/pnp/run_50cross_selectsrc_pilot.sh
```

## Evidence boundary

Scripts that convert or generate files are not by themselves task-success evidence. Accepted generated demonstrations require full simulator rollout, final success, persistent success through the end, post-hold stability, and visual/trace evidence.


## Optional Franka datagen HDF5 input

`select_pnp_50_source_pool.py` keeps the original MolmoBot shard behavior by default. Pass `--franka-datagen-root PATH` to recursively select strict-success trajectories from locally generated `house_*/trajectories_batch_*.h5` files instead:

```bash
PNP_SELECT_N=50 python src/pnp/select_pnp_50_source_pool.py \
  --franka-datagen-root /path/to/datagen/pick_and_place_planner_v1
```

The resulting manifest is compatible with `collect_datagen_info_50cross.py` and `convert_seed_set_to_robomimic_50cross.py`. Franka selection requires terminal/persistent success, phases `0..9`, terminal task success, required replay fields, and unique initial-state/action fingerprints. These are synthetic scripted-IK planner expert demos, not human demonstrations or RB-Y1 planner-server trajectories.

## Fixed-pool source demo conversion

Raw MolmoSpaces `trajectories_batch_*.h5` files are not directly consumable by MimicGen. Build a separate robomimic source HDF5 with one `data/demo_*` group per replay-verified source trajectory:

```bash
export MOLMOSPACES_ROOT="$PWD"
export MOLMOSPACES_PNP_WORKDIR="$PWD/runtime/fixedpool_potato_bowl_1716_source"
export MOLMOSPACES_PYTHON=/path/to/molmospaces-python
export MIMICGEN_ROOT="$PWD/vendor/mimicgen"
export ROBOMIMIC_ROOT="$PWD/vendor/robomimic"
export MUJOCO_GL=egl

bash src/pnp/run_fixedpool_source_hdf5.sh \
  /path/to/datagen/pick_and_place_planner_v1 43
```

The entrypoint defaults to `PNP_FIXEDPOOL_HOUSE_ID=1716` and `PNP_FIXEDPOOL_RUN_NAME_PREFIX=potato_bowl_1716_seed`; set both explicitly for a different fixed pool. It reads raw trajectory HDF5 without modifying it, selects terminal-success candidates, deduplicates exact initial-state/action fingerprints, and replays every retained source. By default, only replay hard-passes (final success and persistent success to the end) are written to the robomimic HDF5. A final-success trajectory with a transient persistence-flag drop may be included only with explicit manual video review through `--manual-review-exceptions`; the output records these exceptions per demo and at the HDF5 root. The generated source set must retain its fixed-pool identity and the `synthetic_scripted_ik_planner_expert` provenance; it is not a human demonstration dataset.

### Completed `potato_bowl_1716` source set (2026-08-01)

The formal source set uses only the seven run roots matching `potato_bowl_1716_seed*` under house `1716`. It contains 43 exact-fingerprint-distinct source trajectories. Each was deterministically replayed before conversion. The final artifact is runtime-only and intentionally ignored by Git:

```text
runtime/fixedpool_potato_bowl_1716_source/artifacts/seeds/
  robomimic_pnp_fixedpool_manual_review43.hdf5
```

The artifact contains 43 `data/demo_*` groups and 7,582 action samples across the seven formal run roots. Automatic replay hard-pass accepted 40 trajectories. The remaining final-success trajectories at source indices `6`, `19`, and `29` had a transient persistence-flag drop but were accepted after Kunyu reviewed their `exo_camera_1` videos on 2026-08-01. They are marked with `manual_review_exception=true` per demo; root attributes retain both the automatic and manual-review seed-index records. The standard validator passed with `--expected-demos 43`.

To reproduce that approved conversion from the completed replay records:

```bash
python src/pnp/convert_seed_set_to_robomimic_50cross.py \
  --accepted all --manual-review-exceptions 6,19,29 \
  --manifest runtime/fixedpool_potato_bowl_1716_source/artifacts/seeds/pnp_seed_manifest_fixedpool.json \
  --replay-root runtime/fixedpool_potato_bowl_1716_source/artifacts/replay_pnp_exact_fixedpool \
  --out runtime/fixedpool_potato_bowl_1716_source/artifacts/seeds/robomimic_pnp_fixedpool_manual_review43.hdf5

python src/pnp/validate_robomimic_source_hdf5.py \
  --input runtime/fixedpool_potato_bowl_1716_source/artifacts/seeds/robomimic_pnp_fixedpool_manual_review43.hdf5 \
  --expected-demos 43
```

## Fixed-pool MimicGen expansion

`run_fixedpool_mimicgen43_to100.sh` expands the approved 43-demo HDF5 into an
audited generated-rollout pool. It invokes MimicGen's per-subtask source
selection across all 43 source demonstrations for each of the 43 recorded
target initial states, records every attempt in
`runtime/fixedpool_potato_bowl_1716_source/logs/`, and accepts an output only
when it has final simulator success, persistence through a 30-step hold, two
non-empty camera videos, and a unique SHA-256 over `generated_actions.npy`.
Failed and exact-duplicate attempts remain recorded but are excluded from
`accepted.jsonl`. Generated HDF5/video runtime artifacts are not committed.

```bash
export MOLMOSPACES_ROOT="$PWD"
export MOLMOSPACES_PNP_WORKDIR="$PWD/runtime/fixedpool_potato_bowl_1716_source"
export MOLMOSPACES_PYTHON=/path/to/molmospaces-python
export MIMICGEN_ROOT=/path/to/mimicgen
export ROBOMIMIC_ROOT=/path/to/robomimic
export PYTHONPATH="$PWD:$MIMICGEN_ROOT:$ROBOMIMIC_ROOT"
export MUJOCO_GL=egl

TARGET_SUCCESS=100 MAX_ATTEMPTS=240 \
  bash src/pnp/run_fixedpool_mimicgen43_to100.sh
```

The launcher uses MimicGen's `select_src_per_subtask` behavior, so a source
segment is selected at each subtask boundary; an attempt may by chance select
the same source demo more than once. This is recorded through `src_demo_inds`,
but is not artificially rejected because MimicGen does not impose a
multi-source-per-episode requirement. The launcher is resume-safe: set
`RUN_DIR` to an existing run directory to skip named attempts already recorded
there. `summary.json` reports the current attempt, strict-success, duplicate,
and accepted-unique counts. The source HDF5 SHA-256 is recorded with every
attempt to preserve the approved 43-demo provenance in downstream generated
demonstrations.
