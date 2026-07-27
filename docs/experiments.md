# Experiments

## Pick-and-Place Integration

The Pick-and-Place work uses MolmoBot-Data source rollouts as synthetic planner expert seeds, converts selected trajectories into robomimic/MimicGen-compatible HDF5, and replays generated trajectories back in MolmoSpaces.

Primary source sets:

- Heterogeneous 10-source set: `results/pnp_seed_manifest.json`.
- Homogeneous foodlike-to-bowl 10-source set: `results/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json`.

Completed evidence included in this repository:

- `results/robomimic_pnp_10demo_aligned.summary.json`: source HDF5 conversion summary.
- `results/robomimic_pnp_10demo_parse_result.json`: MimicGen parser check on the 10-demo source dataset.
- `results/whole_source_transformfirst_summary.json`: generated rollout summary for the heterogeneous whole-source transform-first run.
- `results/collector_uniform_summary_live.json`: live snapshot of the non-deduplicated collector.
- `results/collector_highyield_dedup_summary_live.json`: live snapshot of the action-deduplicated high-yield collector.

Do not treat these live collector summaries as a final 100-success dataset. The collection was still running at the included snapshot.

## Bimanual YAM

The bimanual YAM scripts are included because they are part of the same MolmoSpaces integration effort. They provide scene reachability diagnostics, browser visualization, and keyboard teleoperation.

Evidence boundary:

- Browser visualization proves camera bridge and frame serving.
- Keyboard teleoperation proves interactive control infrastructure.
- Reachability and tabletop checks prove diagnostics only.
- Scripted source-demo generation is not a substitute for human demonstrations unless the output is explicitly labelled scripted expert data.
