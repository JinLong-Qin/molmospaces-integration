# MolmoSpaces Integration Worklines

This repository is a public MolmoSpaces integration portfolio. Pick-and-Place MimicGen is the main current release path, but it is not the only workline. The related MolmoSpaces work is organized here by evidence level and maturity.

## Included worklines

| Workline | Public location | Status | Evidence boundary |
|---|---|---|---|
| MimicGen Pick-and-Place | `src/pnp/`, `results/`, `docs/experiments.md` | Active / primary | Real MolmoSpaces rollouts and lightweight result manifests; large HDF5/videos are runtime artifacts. |
| 50-demo MimicGen cross-subtask pilot | `src/pnp/*50cross*`, `results/50cross_*` | Diagnostic | Closer to MimicGen cross-demo subtask recombination, but broad random mixing did not yield accepted stable rollouts. |
| Bimanual YAM browser teleoperation | `src/bimanual_yam/` | Infrastructure | Browser visualization and keyboard control bridge; not by itself a successful task demonstration. |
| iTHOR bimanual YAM | `docs/worklines/ithor_bimanual_yam.md` | In progress / research workline | Official iTHOR scene resources + bimanual YAM experiments; formal source-demo success must be reported separately from scene/control checks. |
| Completed custom-scene bimanual source baseline | `docs/worklines/bimanual_yam_source_baseline.md` | Completed evidence package | Custom tabletop/box scene baseline, not an official iTHOR result. |
| MolmoAct2 → MolmoSpaces | `docs/worklines/molmoact2_legacy.md` | Legacy / ended | Adapter/API and partial behavior were established; stable MolmoSpaces task success was not established. |
| Bounded official MolmoSpaces reproduction | `docs/worklines/molmospaces_official_reproduction.md` | Completed bounded reproduction | Bounded entrypoint/assets evidence only; not a complete benchmark reproduction. |

## What is intentionally not committed

The full runtime work area contains large runtime artifacts: official data shards, generated HDF5/H5 files, NPY arrays, MP4/PPM media, long JSONL traces, PID files, locks, and transient logs. Those are not normal source-control files. This repository keeps scripts, public docs, result summaries, manifests, and lightweight traces. Large artifacts should be regenerated from the documented workflow or published separately with Git LFS / release assets when needed.
