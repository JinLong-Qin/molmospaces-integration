# Bimanual YAM Scripts

This directory is the code entrypoint for the bimanual YAM browser/teleoperation and iTHOR bimanual YAM worklines.

Canonical workline READMEs:

- [`docs/worklines/bimanual_yam_browser_teleop/README.md`](../../docs/worklines/bimanual_yam_browser_teleop/README.md)
- [`docs/worklines/ithor_bimanual_yam/README.md`](../../docs/worklines/ithor_bimanual_yam/README.md)
- [`docs/worklines/bimanual_yam_source_baseline/README.md`](../../docs/worklines/bimanual_yam_source_baseline/README.md)

## Scripts

- `browser_viewer.py` — browser camera visualization.
- `browser_keyboard_teleop.py` — keyboard teleoperation bridge.
- `validate_tabletop_initialization.py` — tabletop/iTHOR initialization checks.
- `check_dual_object_reachability.py` — reachability diagnostics for two objects.
- `scripted_bimanual_source_demo.py` — scripted diagnostic route.

## Quick checks

```bash
python src/bimanual_yam/browser_viewer.py --help
python src/bimanual_yam/browser_keyboard_teleop.py --help
python src/bimanual_yam/validate_tabletop_initialization.py --help
python src/bimanual_yam/check_dual_object_reachability.py --help
python src/bimanual_yam/scripted_bimanual_source_demo.py --help
```

Actual simulation requires MolmoSpaces assets and a suitable display/browser environment.

## Evidence boundary

This directory provides infrastructure and diagnostics. Browser visualization, keyboard control, scene initialization, reachability checks, and scripted attempts are separate evidence layers and must not be merged into a formal source-demo success claim.
