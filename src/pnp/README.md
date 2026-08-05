# Active Franka Pick-and-Place Pipeline

This directory contains only the current Franka Droid / Pick-and-Place / in-process scripted-IK MimicGen pipeline. Historical 50-cross experiments, fixed-pool wrappers, single-seed collectors, and superseded code snapshots are under [`archive/pnp/`](../../archive/pnp/).

## Active Modules

```text
select_source_pool.py
  -> replay_source_candidate.py
  -> convert_source_hdf5.py
  -> validate_robomimic_source_hdf5.py
  -> source HDF5

sample_fixedbase_target_manifest.py
  -> validate_fixedbase_target_manifest.py
  -> target manifest

source HDF5 + target manifest
  -> run_generation.py --mode per-subtask
  -> generate_pick_place_rollout.py
  -> accepted.jsonl and runtime artifacts
```

- `select_source_pool.py`: selects strict-success source candidates from an official shard or locally generated Franka datagen HDF5.
- `replay_source_candidate.py`: replays one selected source and records the MimicGen datagen fields.
- `convert_source_hdf5.py`: creates a robomimic/MimicGen source HDF5 from replay-accepted sources.
- `validate_robomimic_source_hdf5.py`: validates source HDF5 structure, alignment, finite arrays, and provenance.
- `sample_fixedbase_target_manifest.py` and `validate_fixedbase_target_manifest.py`: create and check target reset layouts.
- `run_source_hdf5_pipeline.py`: parameterized source selection, replay, conversion, and validation orchestrator.
- `run_generation.py`: parameterized rollout orchestrator. Use `--mode per-subtask` for the official MimicGen source-selection path. `--mode whole-source --diagnostic` is control-only.
- `generate_pick_place_rollout.py`: one real MolmoSpaces rollout; it is the execution primitive used by the generation orchestrator.

## Environment

Run from the repository root after `bash tools/setup_mimicgen_dependency.sh`:

```bash
export MOLMOSPACES_ROOT="$PWD"
export MOLMOSPACES_PYTHON=/path/to/molmospaces-python
export MOLMOSPACES_PNP_WORKDIR="$PWD/runtime/mimicgen_pick_and_place"
export MIMICGEN_ROOT="$PWD/vendor/mimicgen"
export ROBOMIMIC_ROOT="$PWD/vendor/robomimic"
export PYTHONPATH="$PWD:$MIMICGEN_ROOT:$ROBOMIMIC_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
export MOLMOSPACES_NLTK_DATA="$NLTK_DATA"
```

## Current 17-Source Pilot

The active controlled pilot uses the replay-verified 17-unique-source HDF5 and an independent target manifest. Paths, source count, target range, RNG base, and output label are explicit parameters; they are not encoded in new script names.

```bash
"$MOLMOSPACES_PYTHON" src/pnp/run_generation.py \
  --root "$MOLMOSPACES_ROOT" \
  --python "$MOLMOSPACES_PYTHON" \
  --work "$MOLMOSPACES_PNP_WORKDIR" \
  --mode per-subtask \
  --source-count 17 \
  --source-hdf5 /path/to/robomimic_pnp_fixedbase_17unique_replayed.hdf5 \
  --target-manifest /path/to/pnp_target_manifest.json \
  --target-success 10 \
  --target-start 0 --target-end 9 \
  --run-label fixed_bowl_control_pilot
```

The generated data are not accepted merely because a process exits successfully. Each accepted rollout requires real simulator execution, final success, success persistence through the post-hold window, non-empty videos, matching target layout, and action/layout deduplication. The current formal expansion count remains zero until these gates pass.

## Historical Material

- `archive/pnp/legacy_50cross/`: dated 50-demo cross-subtask diagnostic scripts.
- `archive/pnp/legacy_collectors/`: dated wrappers and collectors superseded by the parameterized orchestrators.
- `archive/pnp/code_snapshots/`: code snapshots retained for provenance only.
- `archive/docs/worklines/mimicgen_50cross/`: historical workline README.

Existing `runtime/`, `results/`, and `artifacts/` paths are historical evidence and are intentionally unchanged by this reorganization.
