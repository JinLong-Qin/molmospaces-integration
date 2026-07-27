# 50-demo `select_src_per_subtask` pilot snapshot

This directory stores lightweight evidence for the broad 50-demo cross-subtask MimicGen
pilot run from the 4090 work directory:

```text
logs/50cross_selectsrc_pilot_20260727_182533
```

Source artifacts used by that run:

- Source HDF5: `artifacts/seeds/robomimic_pnp_50demo_crossmix_aligned.hdf5`
- Target manifest: `artifacts/seeds/pnp_seed_manifest_50demo_crossmix.json`
- Demo keys: `demo_0..demo_49`
- MimicGen flags: `--select-src-per-subtask --transform-first-robot-pose --post-hold-steps 30 --save-videos`

Included here:

- `status.tsv`, `summary.tsv`, and `launcher.out`: pilot launch and per-seed status.
- `seed_*.log` / `seed_*.exit_code`: copied lightweight run logs for attempted seeds.
- `generated_outputs/seed_*/generate_result.json`: generation metadata from completed output folders.
- `generated_outputs/seed_*/success_trace.json`: simulator success/contact trace for completed output folders.
- `summary.json`: compact index of exit codes, flags, and remote generated-video paths.

Not included in Git:

- Raw source `.h5` shards under `artifacts/seeds/raw_50demo_crossmix/`.
- The generated source HDF5 `robomimic_pnp_50demo_crossmix_aligned.hdf5`.
- Generated `.mp4` diagnostic videos and `.npy` action arrays under `artifacts/mimicgen_pnp/gen_50cross_selectsrc_seed*/`.

Those files are binary runtime artifacts. The repository keeps the scripts, manifests,
summaries, and logs needed to reproduce or audit the run, while the binary artifacts
should be regenerated locally or distributed separately via release artifacts / Git LFS if
needed.

Interpretation: this was a diagnostic broad-random cross-demo subtask pilot, not a final
success-rate experiment. The completed seeds in this snapshot did not produce accepted
persistent-success rollouts, and video review showed source-compatibility / discontinuity
issues. The next valid path is compatibility-filtered `select_src_per_subtask` generation.
