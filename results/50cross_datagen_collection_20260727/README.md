# 50-demo source-pool datagen-info collection snapshot

This directory stores lightweight logs for the 4090 datagen-info collection and source
HDF5 conversion used to build the broad 50-demo Pick-and-Place source pool.

Remote run locations:

```text
logs/collect_50cross_datagen_parallel_20260727_173229
logs/convert_50cross_hdf5_20260727_1824.out
```

Included here:

- `parallel_summary.txt`: launcher summary for the parallel datagen-info collection.
- `parallel_launcher.out`: launcher stdout/stderr snapshot.
- `convert_50cross_hdf5_20260727_1824.out`: conversion stdout containing the selected source demos and HDF5 summary.

Related repository files:

- `results/pnp_seed_manifest_50demo_crossmix.json`
- `results/pnp_50cross_selected_hardpass_indices.json`
- `results/robomimic_pnp_50demo_crossmix_aligned.summary.json`
- `src/pnp/select_pnp_50_source_pool.py`
- `src/pnp/collect_datagen_info_50cross.py`
- `src/pnp/run_collect_50cross_datagen_parallel.sh`
- `src/pnp/convert_seed_set_to_robomimic_50cross.py`

Binary arrays under `artifacts/replay_pnp_exact_50cross/` are not committed. They are
regenerable from the manifest and official MolmoBot data shard using the scripts above.
