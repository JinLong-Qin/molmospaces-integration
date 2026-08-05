# Archived MimicGen Integration Guide

This document is retained only as a dated record of the earlier MimicGen integration workflow. Its commands, fixed-count launchers, and artifact names describe historical experiments and are not active entrypoints.

The active Franka Droid / Pick-and-Place / in-process scripted-IK pipeline is documented at [`src/pnp/README.md`](../../src/pnp/README.md). It uses the parameterized source-HDF5 and generation orchestrators rather than the historical collectors.

Historical code is retained for audit under:

- [`archive/pnp/legacy_50cross/`](../pnp/legacy_50cross/)
- [`archive/pnp/legacy_collectors/`](../pnp/legacy_collectors/)
- [`archive/pnp/code_snapshots/`](../pnp/code_snapshots/)

Existing `runtime/`, `results/`, and `artifacts/` locations are intentionally unchanged. Historical result paths remain evidence records; they do not imply that the archived commands should be run.
