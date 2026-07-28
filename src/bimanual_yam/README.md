# Bimanual YAM Scripts

This directory is the code entrypoint for the bimanual YAM browser/teleoperation and iTHOR bimanual YAM worklines.

Canonical workline READMEs:

- [`docs/worklines/bimanual_yam_browser_teleop/README.md`](../../docs/worklines/bimanual_yam_browser_teleop/README.md)
- [`docs/worklines/ithor_bimanual_yam/README.md`](../../docs/worklines/ithor_bimanual_yam/README.md)
- [`docs/worklines/bimanual_yam_source_baseline/README.md`](../../docs/worklines/bimanual_yam_source_baseline/README.md)

## Scripts

- `browser_keyboard_teleop.py` — supported browser keyboard teleoperation bridge with strict tabletop initialization.
- `validate_tabletop_initialization.py` — tabletop/iTHOR initialization checks.
- `check_dual_object_reachability.py` — reachability diagnostics for two objects.
- `scripted_bimanual_source_demo.py` — scripted diagnostic route.

## Quick checks

```bash
python src/bimanual_yam/browser_keyboard_teleop.py --help
python src/bimanual_yam/validate_tabletop_initialization.py --help
python src/bimanual_yam/check_dual_object_reachability.py --help
python src/bimanual_yam/scripted_bimanual_source_demo.py --help
```

Recommended local teleoperation command:

```bash
python src/bimanual_yam/browser_keyboard_teleop.py \
  --host 127.0.0.1 \
  --port 8765 \
  --house-index 1 \
  --seed 110 \
  --render-fps 8 \
  --control-hz 25 \
  --input-timeout-ms 400 \
  --initialization-max-attempts 50 \
  --initialization-report runtime/bimanual_yam_initialization_report.json
```

Actual simulation requires MolmoSpaces assets and a suitable display/browser environment.

## Evidence boundary

This directory provides infrastructure and diagnostics. Keyboard control, scene initialization, reachability checks, and scripted attempts are separate evidence layers and must not be merged into a formal source-demo success claim.
