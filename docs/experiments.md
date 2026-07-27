# Experiments

## Pick-and-Place Integration

The Pick-and-Place work uses MolmoBot-Data source rollouts as synthetic planner expert seeds, converts selected trajectories into robomimic/MimicGen-compatible HDF5, and replays generated trajectories back in MolmoSpaces.

Primary source sets:

- Heterogeneous 10-source set: `results/pnp_seed_manifest.json`.
- Homogeneous foodlike-to-bowl 10-source set: `results/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json`.
- Broad 50-demo cross-subtask source pool: `results/pnp_seed_manifest_50demo_crossmix.json`.

Completed evidence included in this repository:

- `results/robomimic_pnp_10demo_aligned.summary.json`: source HDF5 conversion summary.
- `results/robomimic_pnp_10demo_parse_result.json`: MimicGen parser check on the 10-demo source dataset.
- `results/whole_source_transformfirst_summary.json`: generated rollout summary for the heterogeneous whole-source transform-first run.
- `results/pnp_50cross_selected_hardpass_indices.json`: `51` strict replay/datagen-info hard-pass candidates, with `50` selected for the broad source pool.
- `results/robomimic_pnp_50demo_crossmix_aligned.summary.json`: 50-demo source HDF5 summary (`50` demos, `9286` action rows) for MimicGen cross-demo subtask recombination.
- `results/50cross_datagen_collection_20260727/`: lightweight datagen-info collection and source-HDF5 conversion logs.
- `results/50cross_selectsrc_pilot_20260727_182533/`: lightweight `select_src_per_subtask=True` pilot logs, generated-output JSON traces, and a pointer to the remote generated videos.
- `results/collector_uniform_summary_live.json`: live snapshot of the non-deduplicated collector.
- `results/collector_highyield_dedup_summary_live.json`: live snapshot of the action-deduplicated high-yield collector.

The homogeneous 100-pilot used one source demo per generated rollout. The 50-demo route is the closer MimicGen-style experiment: it enables `select_src_per_subtask=True`, so a single generated rollout can choose different source demos for different subtasks. The initial broad random pilot showed that arbitrary mixing across heterogeneous MolmoSpaces houses, objects, and receptacles can create discontinuity, IK/contact issues, and invalid diagnostic videos. Do not treat that broad pilot as a final success-rate estimate; use compatibility-filtered subsets before scaling.

Do not treat these live collector summaries as a final 100-success dataset. The collection was still running at the included snapshot.

## Bimanual YAM

The bimanual YAM scripts are included because they are part of the same MolmoSpaces integration effort. They provide scene reachability diagnostics, browser visualization, and keyboard teleoperation.

Evidence boundary:

- Browser visualization proves camera bridge and frame serving.
- Keyboard teleoperation proves interactive control infrastructure.
- Reachability and tabletop checks prove diagnostics only.
- Scripted source-demo generation is not a substitute for human demonstrations unless the output is explicitly labelled scripted expert data.
