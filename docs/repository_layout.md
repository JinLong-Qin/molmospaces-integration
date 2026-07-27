# Repository Layout

```text
src/pnp/             Pick-and-Place integration scripts
src/bimanual_yam/    bimanual YAM diagnostics and keyboard teleoperation
results/             lightweight JSON manifests and run summaries
docs/                public documentation
```

The GitHub repository intentionally excludes raw experiment ledgers, dated run logs, backup files, HDF5 datasets, videos, PID files, and generated simulator artifacts. Those files belong in local runtime storage or external artifact releases, not in source control.
