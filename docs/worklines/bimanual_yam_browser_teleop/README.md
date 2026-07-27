# Bimanual YAM Browser Teleoperation Workline

## Purpose

This workline provides human-facing browser visualization and keyboard teleoperation infrastructure for bimanual YAM experiments in MolmoSpaces.

It is a tooling/infrastructure workline, not a standalone task-success result.

## Public code

Code directory: [`src/bimanual_yam/`](../../../src/bimanual_yam/)

Key scripts:

- `browser_viewer.py` — serves top/left/right camera views for browser inspection.
- `browser_keyboard_teleop.py` — maps keyboard input to bimanual control commands.
- `validate_tabletop_initialization.py` — validates tabletop / scene initialization assumptions.
- `check_dual_object_reachability.py` — checks whether both target objects are reachable under a layout.
- `scripted_bimanual_source_demo.py` — scripted diagnostic route for bimanual source-demo attempts.

## Minimal run sequence

After standard setup:

```bash
python src/bimanual_yam/browser_viewer.py --help
python src/bimanual_yam/browser_keyboard_teleop.py --help
python src/bimanual_yam/validate_tabletop_initialization.py --help
python src/bimanual_yam/check_dual_object_reachability.py --help
python src/bimanual_yam/scripted_bimanual_source_demo.py --help
```

Actual simulator runs require MolmoSpaces assets and a display/browser-capable environment.

## Public evidence

Related runtime inventory: [`results/workline_index/ithor_bimanual_yam.md`](../../../results/workline_index/ithor_bimanual_yam.md).

## Evidence boundary

Valid claim: the repository contains browser visualization, keyboard teleoperation, initialization, reachability, and scripted diagnostic utilities for bimanual YAM experiments.

Invalid claim: browser control, camera visibility, or reachability checks alone do not prove a successful bimanual source demonstration.
