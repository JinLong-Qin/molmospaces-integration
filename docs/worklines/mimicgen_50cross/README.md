# 50-demo MimicGen Cross-Subtask Workline

## Purpose

This diagnostic workline tests a route closer to MimicGen's original cross-demo subtask recombination idea: generated rollouts may select different source demonstrations per subtask by using `select_src_per_subtask=True`.

It is separate from the whole-source Pick-and-Place route, where one generated rollout follows one source demonstration after object-centric transformation.

## Public code

The scripts live in [`src/pnp/`](../../../src/pnp/):

- `select_pnp_50_source_pool.py`
- `collect_datagen_info_50cross.py`
- `convert_seed_set_to_robomimic_50cross.py`
- `run_collect_50cross_datagen.sh`
- `run_collect_50cross_datagen_parallel.sh`
- `run_collect_50cross_extra_workers.sh`
- `run_50cross_selectsrc_pilot.sh`

## Minimal run sequence

After standard setup and official-shard placement:

```bash
python src/pnp/select_pnp_50_source_pool.py --help
bash src/pnp/run_collect_50cross_datagen.sh
python src/pnp/convert_seed_set_to_robomimic_50cross.py --help
bash src/pnp/run_50cross_selectsrc_pilot.sh
```

The shell scripts are parameterized by environment variables, especially `MOLMOSPACES_ROOT`, `MOLMOSPACES_PNP_WORKDIR`, `MOLMOSPACES_PYTHON`, `MIMICGEN_ROOT`, and `ROBOMIMIC_ROOT`.

## Public evidence

Committed lightweight evidence:

- [`results/pnp_50cross_selected_hardpass_indices.json`](../../../results/pnp_50cross_selected_hardpass_indices.json)
- [`results/pnp_seed_manifest_50demo_crossmix.json`](../../../results/pnp_seed_manifest_50demo_crossmix.json)
- [`results/robomimic_pnp_50demo_crossmix_aligned.summary.json`](../../../results/robomimic_pnp_50demo_crossmix_aligned.summary.json)
- [`results/50cross_datagen_collection_20260727/`](../../../results/50cross_datagen_collection_20260727/)
- [`results/50cross_selectsrc_pilot_20260727_182533/`](../../../results/50cross_selectsrc_pilot_20260727_182533/)

## Evidence boundary

Valid claim: the repository includes a 50-demo cross-subtask MimicGen diagnostic route, including source-pool selection, datagen-info collection, HDF5 summary, and bounded select-src-per-subtask pilot traces.

Invalid claim: do not report the broad random 50-demo cross-subtask pilot as a successful generated dataset. The committed pilot evidence records diagnostic failures under broad random mixing and motivates compatibility-filtered source/target subsets.
