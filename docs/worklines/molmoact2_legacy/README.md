# MolmoAct2 → MolmoSpaces Legacy Workline

## Purpose

This workline connected MolmoAct2-BimanualYAM to MolmoSpaces and tested whether the official MolmoAct2 YAM policy could run closed-loop through a MolmoSpaces adapter.

It is included because it is part of the MolmoSpaces integration portfolio, but it is a legacy/ended diagnostic line rather than the current public success claim.

## Public code

Code directory: [`src/molmoact2_legacy/`](../../../src/molmoact2_legacy/)

Main contents:

- `overlay/` — adapter/config/policy/view files that were used as MolmoSpaces extensions.
- `scripts/` — diagnostic, replay, smoke, frame-probe, and bridge scripts.
- `tests/` — MolmoAct2-specific component tests and fixtures.
- `docs/prd_20260623/` — historical PRD/config snapshots.
- `README_public.md` — public-facing summary of the legacy line.

## Minimal run sequence

This workline is archived. It should not be copied back into the live MolmoSpaces tree piecemeal. For inspection after clone:

```bash
python -m py_compile $(find src/molmoact2_legacy -name '*.py' -type f | sort)
python src/molmoact2_legacy/scripts/smoke_molmoact2_yam_cli_selection.py --help
python src/molmoact2_legacy/scripts/smoke_molmoact2_yam_policy.py --help
```

Full replay requires the original model/runtime assets and should be reconstructed in a separate checkout or branch.

## Public evidence

Inventory: [`results/workline_index/molmoact2_integration_legacy.md`](../../../results/workline_index/molmoact2_integration_legacy.md)

## Evidence boundary

Valid claim: MolmoAct2 `/act` plumbing, three-camera packets, simulator stepping, action decoding, HDF5/MP4 artifact generation, and partial approach/contact/grasp/lift behavior were explored through a MolmoSpaces adapter. Official MolmoAct2 SAPIEN `sim_eval` produced a genuine successful episode in its own environment.

Invalid claim: do not claim stable complete MolmoSpaces task success for MolmoAct2-BimanualYAM. The official SAPIEN success and the MolmoSpaces adapter diagnostics are separate evidence layers.
