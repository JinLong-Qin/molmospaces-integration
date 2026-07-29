# MimicGen Single-Arm Groundwork Workline

## Purpose

This historical workline explored single-arm MolmoSpaces trajectory replay, alignment, and seed conversion before the public Pick-and-Place integration was consolidated under `src/pnp/`.

It remains important because it explains the earlier replay/alignment evidence behind the later PnP pipeline, but it is not the current primary release path.

## Public code

The maintained public code for this family of work is now in [`src/pnp/`](../../../src/pnp/). The historical runtime scripts are represented in the inventory rather than copied wholesale into a parallel source tree.

## Minimal run sequence

For a fresh clone, use the Pick-and-Place scripts instead of the historical single-arm scripts:

This historical workline is not the official RB-Y1 scripted/planner datagen entrypoint. If you are trying to run single-arm planner expert generation with `RBY1PickDataGenConfig` or `RBY1PickAndPlaceDataGenConfig`, use `python -m molmo_spaces.data_generation.main ...`, make sure `curobo` is installed, and verify planner-server connectivity for your environment; do not treat the `src/pnp/` scripts here as a drop-in replacement for that pipeline.

```bash
python src/pnp/replay_source_episode.py --help
python src/pnp/convert_single_seed_to_robomimic.py --help
python src/pnp/parse_source_dataset.py --help
```

## Public evidence

Inventory: [`results/workline_index/mimicgen_single_arm.md`](../../../results/workline_index/mimicgen_single_arm.md)

The inventory records replay summaries, step traces, seed manifests, logs, and historical scripts from the runtime work area. Large videos, NPY files, and generated datasets are not committed in normal Git.

## Evidence boundary

Valid claim: this workline provided single-arm replay/alignment groundwork for the later MimicGen Pick-and-Place integration.

Invalid claim: do not present it as the current formal PnP generated-rollout result or as a complete MimicGen reproduction.
