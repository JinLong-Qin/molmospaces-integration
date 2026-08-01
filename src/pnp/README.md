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

## Main script groups

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

- `generate_pick_place_rollout.py`
- `collect_uniform_successes.sh`
- `collect_unique_highyield_successes.sh`

### 50-demo cross-subtask diagnostics

- `select_pnp_50_source_pool.py`
- `collect_datagen_info_50cross.py`
- `convert_seed_set_to_robomimic_50cross.py`
- `run_collect_50cross_datagen.sh`
- `run_collect_50cross_datagen_parallel.sh`
- `run_collect_50cross_extra_workers.sh`
- `run_50cross_selectsrc_pilot.sh`

### Fixed-pool source-HDF5 build

- `run_fixedpool_source_hdf5.sh` - selects, deduplicates, replays, converts, and validates a fixed-pool source set.
- `validate_robomimic_source_hdf5.py` - verifies the generated `data/demo_*` HDF5 structure, alignment, finite numeric arrays, and source provenance.

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
