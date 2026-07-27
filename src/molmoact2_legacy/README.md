# Legacy MolmoAct2 → MolmoSpaces Integration

## State

This workline is ended and isolated from the live MolmoSpaces tree. On 2026-07-22, 6 MolmoAct2 root processes and 6 child processes were stopped cleanly with `SIGTERM`; no matching process or listener on ports 8202/8203 remained.

## Contents

- `overlay/` — untracked package extensions removed from live `molmo_spaces/`
- `scripts/` — 23 historical diagnostic/bridge scripts
- `tests/` — MolmoAct2-specific tests and fixtures
- `examples/` — four MolmoAct2 scene/example trees
- `logs/` — historical run logs
- `artifacts/` — all `molmoact2_*` evidence directories
- `tool_state/` — Ralph historical state
- `backups/` — source/script backups and their prior manifest
- `snapshot/tracked_changes.patch` — complete patch for the seven formerly modified tracked files
- `snapshot/tracked_modified_files/` — exact pre-restore copies
- `snapshot/files_before_migration.json` and `final_manifest.json` — provenance inventories

## Live-tree result

The seven tracked files were restored to base commit `fb49aef0750973bff8660e213612d06005f8a37f`; MolmoAct2 overlay files, scripts, examples, tests, logs, and artifacts were removed from the live code paths.

## Evidence boundary

This archive is not current infrastructure and must not be imported by current iTHOR work. Any reconstruction should happen in a separate checkout using the frozen patch and manifests, never by copying files back piecemeal.
