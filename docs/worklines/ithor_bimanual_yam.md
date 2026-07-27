# iTHOR Bimanual YAM Workline

## Summary

This workline uses official MolmoSpaces iTHOR scene resources and the upstream bimanual YAM embodiment to build and validate bimanual source-demo infrastructure. It is separate from the Pick-and-Place MimicGen workline and from the older MolmoAct2 adapter.

## Public components in this repository

- `src/bimanual_yam/browser_viewer.py`: browser camera visualization.
- `src/bimanual_yam/browser_keyboard_teleop.py`: loopback browser keyboard teleoperation.
- `src/bimanual_yam/validate_tabletop_initialization.py`: tabletop/iTHOR initialization validation.
- `src/bimanual_yam/check_dual_object_reachability.py`: dual-object reachability diagnostics.
- `src/bimanual_yam/scripted_bimanual_source_demo.py`: scripted source-demo diagnostic route.

## Evidence boundary

Scene loading, camera setup, reachability checks, browser control, and scripted diagnostics are separate evidence layers. They do not by themselves prove a formal source-demonstration dataset or a successful human demonstration.

## Relation to other worklines

- The completed custom-scene source baseline is documented separately.
- The MolmoAct2 adapter line is legacy and should not be mixed into current iTHOR claims.
- Pick-and-Place MimicGen is a single-arm/Franka PnP data-generation line, not this bimanual iTHOR/YAM line.
