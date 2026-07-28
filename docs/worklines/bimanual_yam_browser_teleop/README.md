# Bimanual YAM Browser Teleoperation Workline

## Purpose

This workline provides human-facing browser visualization and keyboard teleoperation infrastructure for bimanual YAM experiments in MolmoSpaces.

It is a tooling/infrastructure workline, not a standalone task-success result.

## Environment requirements

- **GPU rendering is required** for usable frame rates. CPU software rendering (OSMesa) with three cameras gives ~3 FPS.
- **NVIDIA EGL headless rendering** is the tested configuration: Linux + NVIDIA GPU + proprietary driver + EGL vendor (`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`). The teleop automatically uses `MUJOCO_GL=egl` via MolmoSpaces.
- **WSL2 is not supported**. WSL2 uses Mesa EGL, which does not expose `EGL_EXT_platform_device`.

## Public code

Code directory: [`src/bimanual_yam/`](../../../src/bimanual_yam/)

Key scripts:

- `browser_keyboard_teleop.py` — supported public entrypoint; performs strict tabletop initialization and maps keyboard input to bimanual control commands.
- `validate_tabletop_initialization.py` — validates tabletop / scene initialization assumptions.
- `check_dual_object_reachability.py` — checks whether both target objects are reachable under a layout.
- `scripted_bimanual_source_demo.py` — scripted diagnostic route for bimanual source-demo attempts.

## Minimal run sequence

### Environment requirements

- **GPU rendering is required**. CPU rendering (OSMesa) with three cameras gives ~3 FPS.
- **NVIDIA EGL headless rendering** is the tested configuration: Linux + NVIDIA GPU + proprietary driver + `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
- **WSL2 is not supported**. WSL2 Mesa EGL does not expose `EGL_EXT_platform_device`.
- **Remote GPU server**: set up an SSH tunnel before opening the browser:
  ```bash
  # On your local machine:
  ssh -L 8765:127.0.0.1:8765 user@your-gpu-server
  ```

Inspect script options:

```bash
python src/bimanual_yam/browser_keyboard_teleop.py --help
python src/bimanual_yam/validate_tabletop_initialization.py --help
python src/bimanual_yam/check_dual_object_reachability.py --help
python src/bimanual_yam/scripted_bimanual_source_demo.py --help
```

Recommended teleoperation command:

```bash
HF_HOME=/mnt/vqa/.cache/huggingface \
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

Set `HF_HOME` to wherever the CLIP model is cached on your machine. On a fresh environment with a SOCKS proxy, install `httpx[socks]` first (`pip install httpx[socks]`) or the CLIP weight download will fail.

Open `http://127.0.0.1:8765` after the terminal prints the teleoperation URL.

## Public evidence

Related runtime inventory: [`results/workline_index/ithor_bimanual_yam.md`](../../../results/workline_index/ithor_bimanual_yam.md).

## Evidence boundary

Valid claim: the repository contains keyboard teleoperation, initialization, reachability, and scripted diagnostic utilities for bimanual YAM experiments.

Invalid claim: browser control, camera visibility, or reachability checks alone do not prove a successful bimanual source demonstration.
