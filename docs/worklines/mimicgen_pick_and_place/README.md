# MimicGen Pick-and-Place Workline

## Purpose

This is the primary active public workline. It connects MolmoSpaces Pick-and-Place rollouts to the MimicGen / robomimic data format, then executes generated trajectories back in the MolmoSpaces simulator.

## Public code

Main code directory: [`src/pnp/`](../../../src/pnp/)

Key scripts:

- `inspect_source_candidates.py` — inspect candidate source demonstrations from the official MolmoBot shard.
- `select_pick_place_seeds.py` — select a source set.
- `replay_source_episode.py` — replay a source episode in MolmoSpaces.
- `collect_datagen_info.py` and `collect_homogeneous_datagen_info.py` — collect MimicGen datagen information.
- `convert_seed_set_to_robomimic.py` and `convert_single_seed_to_robomimic.py` — convert source trajectories to robomimic-style HDF5.
- `parse_source_dataset.py` — parse the source HDF5 with MimicGen compatibility checks.
- `generate_pick_place_rollout.py` — run a generated trajectory back in MolmoSpaces.
- `collect_uniform_successes.sh` and `collect_unique_highyield_successes.sh` — collector entrypoints used for larger sweeps.

## Minimal run sequence

After the standard clone setup in [`../README.md`](../README.md), place the official MolmoBot Pick-and-Place shard under `runtime/`. Then use the top-level README workflow or run the scripts in this order:

This workline is the public `src/pnp/` integration path. It is not the same entrypoint as the official RB-Y1 scripted/planner data-generation configs under `python -m molmo_spaces.data_generation.main ...`. If you need `RBY1PickDataGenConfig` or `RBY1PickAndPlaceDataGenConfig`, follow the dedicated CuRobo installation notes in the top-level [`README.md`](../../../README.md) instead of assuming the `src/pnp/` setup is sufficient.

```bash
python src/pnp/inspect_source_candidates.py --help
python src/pnp/select_pick_place_seeds.py --help
python src/pnp/replay_source_episode.py --help
python src/pnp/collect_datagen_info.py --help
python src/pnp/convert_seed_set_to_robomimic.py --help
python src/pnp/parse_source_dataset.py --help
python src/pnp/generate_pick_place_rollout.py --help
```

The exact data paths depend on where the official shard is placed. The scripts use environment variables such as `MOLMOSPACES_ROOT`, `MOLMOSPACES_PNP_WORKDIR`, `MIMICGEN_ROOT`, and `ROBOMIMIC_ROOT` instead of private machine paths.

## Public evidence

Lightweight committed evidence includes source manifests, HDF5 summaries, parser summaries, and generated rollout traces under `results/`. The full runtime inventory is indexed at [`results/workline_index/mimicgen_pick_and_place.md`](../../../results/workline_index/mimicgen_pick_and_place.md).

## Evidence boundary

Valid claim: this repository contains a reproducible MolmoSpaces Pick-and-Place integration path for source replay, MimicGen-format conversion, datagen-info extraction, and generated rollout execution.

Invalid claim: do not treat a summary file or HDF5 conversion alone as task success. Accepted generated demonstrations require full simulator rollout, final success, success persistence to the end, a post-hold stability window, and non-empty video/trace evidence.


## Newly generated Franka HDF5 input

The source selector can now consume locally generated Franka Pick-and-Place HDF5 while retaining the original MolmoBot shard input:

```bash
PNP_SELECT_N=50 python src/pnp/select_pnp_50_source_pool.py \
  --franka-datagen-root /path/to/datagen/pick_and_place_planner_v1
```

Franka mode recursively reads `house_*/trajectories_batch_*.h5`, requires complete phases `0..9`, terminal/persistent task success and the replay fields used downstream, and rejects duplicate initial-state/action fingerprints. Its manifest remains compatible with the existing replay, datagen-info, robomimic conversion, and MimicGen generation path. Treat `samples_per_house` only as a requested workload and verify actual HDF5 success counts. The source provenance is synthetic scripted-IK planner expert, not human or RB-Y1 planner-server data.
