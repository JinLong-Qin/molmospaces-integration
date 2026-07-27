# Legacy MolmoAct2-to-MolmoSpaces Code Snapshot

This directory contains the public, lightweight part of the MolmoAct2-to-MolmoSpaces integration workline:

- `overlay/`: adapter/config/policy code that was previously tested outside the live upstream tree;
- `scripts/`: diagnostic, smoke, and replay scripts;
- `tests/`: component tests and fixtures for adapter behavior and visual/physics invariants;
- `docs/prd_20260623/`: early alignment design notes.

Evidence boundary: this is a legacy diagnostic integration. It demonstrates adapter/API plumbing and partial behavior probes, but it should not be described as a stable successful MolmoSpaces reproduction of MolmoAct2-BimanualYAM.

Large videos, frames, HDF5 files, action traces, raw logs, process state, local backups, and private machine paths are intentionally not committed here.
