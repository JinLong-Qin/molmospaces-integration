# Bimanual YAM Browser Teleoperation Workline

## Purpose

This workline provides human-facing browser visualization and keyboard teleoperation infrastructure for bimanual YAM experiments in MolmoSpaces.

It is a tooling/infrastructure workline, not a standalone task-success result.

## Public code

Code directory: [`src/bimanual_yam/`](../../../src/bimanual_yam/)

Key scripts:

- `browser_keyboard_teleop.py` — supported public entrypoint; performs strict tabletop initialization and maps keyboard input to bimanual control commands.
- `validate_tabletop_initialization.py` — validates tabletop / scene initialization assumptions.
- `check_dual_object_reachability.py` — checks whether both target objects are reachable under a layout.
- `scripted_bimanual_source_demo.py` — scripted diagnostic route for bimanual source-demo attempts.

## Minimal run sequence

After standard setup, inspect script options:

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

Open `http://127.0.0.1:8765` after the terminal prints the local teleoperation URL.

Actual simulator runs require MolmoSpaces assets and a display/browser-capable environment.

## Public evidence

Related runtime inventory: [`results/workline_index/ithor_bimanual_yam.md`](../../../results/workline_index/ithor_bimanual_yam.md).

## Evidence boundary

Valid claim: the repository contains keyboard teleoperation, initialization, reachability, and scripted diagnostic utilities for bimanual YAM experiments.

Invalid claim: browser control, camera visibility, or reachability checks alone do not prove a successful bimanual source demonstration.
