# Pick-and-Place Augmentation Pipeline

This directory is the supported Python implementation of the reusable
MolmoSpaces × MimicGen Pick-and-Place data-augmentation workflow. Public
experiments are configuration-driven and scenario parameters must be supplied
explicitly; active source files do not encode a private machine path, a fixed
scene, or a particular object pair.

**Start here:** [`docs/pnp-data-augmentation.md`](../../docs/pnp-data-augmentation.md).

## Architecture

```text
validated source HDF5 ─┐
                       ├─ run_experiment.py + JSON configuration
validated target manifest ┘            │
                                       ▼
                              run_generation.py
                                       ▼
                     generate_pick_place_rollout.py
                                       ▼
       per-attempt artifacts + JSONL provenance + HDF5 + summary
```

| Module | Responsibility |
| --- | --- |
| `select_source_pool.py` | Select source candidates from supported source data. |
| `replay_source_candidate.py` | Replay a source candidate and record MimicGen datagen fields. |
| `convert_source_hdf5.py` | Convert replay-accepted sources into a MimicGen/robomimic HDF5. |
| `validate_robomimic_source_hdf5.py` | Validate source schema, finite values, alignment, and provenance. |
| `sample_fixedbase_target_manifest.py` | Sample reset-only target `EpisodeSpec` records from explicit scenario arguments. |
| `validate_fixedbase_target_manifest.py` | Verify target-manifest schema, uniqueness, and scenario consistency. |
| `run_source_hdf5_pipeline.py` | Orchestrate source selection, replay, conversion, and validation. |
| `run_experiment.py` | Validate a portable JSON generation configuration and invoke the batch runner. |
| `run_generation.py` | Batch orchestration, input hashes, deduplication, JSONL records, and summary. |
| `generate_pick_place_rollout.py` | Single real simulator rollout primitive. |

## Boundaries

- Python contains data, simulator, validation, and orchestration logic.
- `scripts/pnp/` is the shell boundary: it only locates the repository and
  interpreter, then invokes Python. It contains no Python heredoc or dataset
  policy.
- `configs/pnp/` stores portable example configurations. Copy an example for a
  run; never encode a local path or scenario in a tracked launcher.
- `archive/pnp/` contains historical one-off experiments and debug snapshots.
  They are preserved for provenance and are not supported APIs.

## Public entrypoints

```bash
# Inspect all CLI contracts first.
"$MOLMOSPACES_PYTHON" src/pnp/run_source_hdf5_pipeline.py --help
"$MOLMOSPACES_PYTHON" src/pnp/sample_fixedbase_target_manifest.py --help
"$MOLMOSPACES_PYTHON" src/pnp/validate_fixedbase_target_manifest.py --help
"$MOLMOSPACES_PYTHON" src/pnp/run_experiment.py --help

# Validate a copied configuration without starting a rollout.
scripts/pnp/run_generation.sh configs/pnp/generation.example.json --dry-run
```

The example intentionally references placeholder input artifacts. It is a
configuration/schema smoke test, not a runnable dataset collection command.
Use a validated source HDF5 and target manifest to run a real smoke.

## Evidence contract

Successful process completion is insufficient. `run_generation.py` records an
attempt only as accepted when task behavior, requested target, selection mode,
post-hold persistence, nonempty media, HDF5 persistence, and action/layout
uniqueness all pass. Dataset-level eligibility additionally requires the target
acceptance count and generated aggregate HDF5. See the full artifact contract
and reproducibility requirements in the workflow guide.
