# iTHOR Bimanual YAM Workline

## Purpose

This workline uses official-style iTHOR / ProcTHOR scene resources with the bimanual YAM embodiment in MolmoSpaces. It is distinct from both Pick-and-Place MimicGen and the older MolmoAct2 legacy adapter.

## Public code

The reusable public scripts are shared with the bimanual browser/teleop workline under [`src/bimanual_yam/`](../../../src/bimanual_yam/):

- `browser_keyboard_teleop.py`
- `validate_tabletop_initialization.py`
- `check_dual_object_reachability.py`
- `scripted_bimanual_source_demo.py`

## Minimal run sequence

Use the standard clone setup, make sure MolmoSpaces assets are available, then inspect the script options:

```bash
python src/bimanual_yam/validate_tabletop_initialization.py --help
python src/bimanual_yam/check_dual_object_reachability.py --help
python src/bimanual_yam/browser_keyboard_teleop.py --help
python src/bimanual_yam/scripted_bimanual_source_demo.py --help
```

For browser teleoperation, use the keyboard bridge directly:

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

The original runtime workline included many simulator artifacts and diagnostic logs. Those are not committed as raw files; they are indexed in the inventory below.

## Public evidence

Inventory: [`results/workline_index/ithor_bimanual_yam.md`](../../../results/workline_index/ithor_bimanual_yam.md)

The inventory records scene/camera/control/reachability/scripted diagnostic artifacts from the runtime work area while keeping raw videos, HDF5/H5 files, PID files, and large logs out of normal Git.

## Evidence boundary

Valid claim: this workline contains official-style iTHOR scene setup, bimanual YAM camera/control infrastructure, layout and reachability checks, and scripted diagnostic attempts.

Invalid claim: scene loading, camera correctness, browser control, reachability, or scripted diagnostics are not the same as a formal accepted bimanual source-demo dataset. Formal source-demo success must be reported separately with strict replay/QA/video evidence.
