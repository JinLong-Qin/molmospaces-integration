# Pick-and-Place MimicGen Scripts

This directory is the code entrypoint for the MimicGen Pick-and-Place worklines.

Canonical workline READMEs:

- [`docs/worklines/mimicgen_pick_and_place/README.md`](../../docs/worklines/mimicgen_pick_and_place/README.md)
- [`docs/worklines/mimicgen_50cross/README.md`](../../docs/worklines/mimicgen_50cross/README.md)

## Standard environment variables

```bash
export MOLMOSPACES_ROOT="$PWD"
export MOLMOSPACES_PNP_WORKDIR="$PWD/runtime/pnp"
export MIMICGEN_ROOT="$PWD/third_party/mimicgen"
export ROBOMIMIC_ROOT="$PWD/third_party/robomimic"
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

## Quick checks

```bash
python src/pnp/parse_source_dataset.py --help
python src/pnp/generate_pick_place_rollout.py --help
bash -n src/pnp/run_50cross_selectsrc_pilot.sh
```

## Evidence boundary

Scripts that convert or generate files are not by themselves task-success evidence. Accepted generated demonstrations require full simulator rollout, final success, persistent success through the end, post-hold stability, and visual/trace evidence.
