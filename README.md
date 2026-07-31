<div align="center">
  <img src="media/banner.svg" alt="MolmoSpaces Integration" width="980" />
</div>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <strong>MolmoSpaces Integration</strong> — Pick-and-Place MimicGen trajectory generation, bimanual YAM teleoperation, and other integration worklines on top of MolmoSpaces.
</p>

<p align="center">
  <img src="media/gif/heterogeneous_generated_examples.gif" alt="Generated rollout examples" width="245" />
  &nbsp;
  <img src="media/gif/source_candidate_example.gif" alt="Source candidate example" width="245" />
  &nbsp;
  <img src="media/gif/foodlike_pilot_outcomes.gif" alt="Pilot outcomes" width="245" />
</p>

<p align="center">
  <a href="media/heterogeneous_generated_examples.mp4">Generated examples</a>
  &nbsp;·&nbsp;
  <a href="media/foodlike_source_candidates_grid.mp4">Source candidates</a>
  &nbsp;·&nbsp;
  <a href="media/foodlike_pilot_outcomes.mp4">Pilot outcomes</a>
</p>

<p align="center"><em>Previews are compact; open the MP4 links for full-size demos.</em></p>

## What is included

This repository is a MolmoSpaces integration workline portfolio. The top-level README lists every public workline so a new reader does not need to browse subdirectories to discover what is here.

| Workline | Canonical README | Code entrypoint | Evidence / inventory | Status |
|---|---|---|---|---|
| MimicGen Pick-and-Place | [`docs/worklines/mimicgen_pick_and_place/README.md`](docs/worklines/mimicgen_pick_and_place/README.md) | [`src/pnp/`](src/pnp/) | [`results/workline_index/mimicgen_pick_and_place.md`](results/workline_index/mimicgen_pick_and_place.md) | Active / primary |
| 50-demo MimicGen cross-subtask route | [`docs/worklines/mimicgen_50cross/README.md`](docs/worklines/mimicgen_50cross/README.md) | [`src/pnp/*50cross*`](src/pnp/) | [`results/50cross_selectsrc_pilot_20260727_182533/`](results/50cross_selectsrc_pilot_20260727_182533/) | Diagnostic |
| Bimanual YAM browser teleoperation | [`docs/worklines/bimanual_yam_browser_teleop/README.md`](docs/worklines/bimanual_yam_browser_teleop/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | Infrastructure |
| iTHOR bimanual YAM | [`docs/worklines/ithor_bimanual_yam/README.md`](docs/worklines/ithor_bimanual_yam/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | In progress |
| Completed custom-scene bimanual YAM baseline | [`docs/worklines/bimanual_yam_source_baseline/README.md`](docs/worklines/bimanual_yam_source_baseline/README.md) | Inventory / regeneration entrypoints | [`results/workline_index/bimanual_yam_source_baseline.md`](results/workline_index/bimanual_yam_source_baseline.md) | Completed evidence package |
| MolmoAct2 → MolmoSpaces legacy | [`docs/worklines/molmoact2_legacy/README.md`](docs/worklines/molmoact2_legacy/README.md) | [`src/molmoact2_legacy/`](src/molmoact2_legacy/) | [`results/workline_index/molmoact2_integration_legacy.md`](results/workline_index/molmoact2_integration_legacy.md) | Legacy / ended |
| Bounded official MolmoSpaces reproduction | [`docs/worklines/molmospaces_official_reproduction/README.md`](docs/worklines/molmospaces_official_reproduction/README.md) | Upstream MolmoSpaces entrypoints | [`results/workline_index/molmospaces_official_reproduction.md`](results/workline_index/molmospaces_official_reproduction.md) | Completed bounded reproduction |
| MimicGen single-arm groundwork | [`docs/worklines/mimicgen_single_arm/README.md`](docs/worklines/mimicgen_single_arm/README.md) | Superseded by [`src/pnp/`](src/pnp/) | [`results/workline_index/mimicgen_single_arm.md`](results/workline_index/mimicgen_single_arm.md) | Historical groundwork |

Also included: the upstream MolmoSpaces source snapshot, lightweight manifests and summaries under `results/`, small README demo media under `media/`, public documentation, and attribution files.

Large runtime data are excluded from Git: official shards, generated HDF5 files, rollout directories, simulator logs, PID files, caches, videos, local machine paths, and internal planning ledgers. Worklines that use such files keep a README, lightweight inventory, or regeneration entrypoint instead of committing the raw artifact.

## Primary Reproduction Path

The primary workflow in this repository is:

> Franka Pick-and-Place datagen -> validated HDF5/video artifacts -> MimicGen source conversion and rollout generation

The command-line value `--robot droid` selects a **Franka robot with DROID-style cameras**. It does not select an RB-Y1 robot. The RB-Y1 CuRobo/planner-server pipeline is a separate optional upstream workline and is not required for the Franka workflow below. See [`docs/worklines/molmospaces_official_reproduction/README.md`](docs/worklines/molmospaces_official_reproduction/README.md) only if that separate workline is your goal.

## Franka Datagen Quick Start

### 1. Clone

```bash
git clone https://github.com/yanggoumao2/molmospaces-integration.git
cd molmospaces-integration
```

SSH also works:

```bash
git clone git@github.com:yanggoumao2/molmospaces-integration.git
cd molmospaces-integration
```

### 2. Create a Python environment

MolmoSpaces uses Python 3.11.

With conda:

```bash
conda create -n molmospaces-integration python=3.11
conda activate molmospaces-integration
python -m pip install --upgrade pip setuptools wheel
```

With `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
```

### 3. Install the Franka datagen dependencies

Install MolmoSpaces with the MuJoCo extra. Franka datagen does not require CuRobo or an RB-Y1 planner server.

```bash
pip install -e ".[mujoco]"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

If a SOCKS proxy is configured and `httpx` reports that `socksio` is missing, install:

```bash
pip install httpx[socks]
```

Set persistent model and language-resource cache locations before the first run:

```bash
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export NLTK_DATA=${NLTK_DATA:-$HOME/nltk_data}
export MOLMOSPACES_NLTK_DATA=$NLTK_DATA
```

### 4. Verify the base installation

```bash
python - <<'PY'
import mujoco
import torch
import warp
import molmo_spaces

print("MolmoSpaces:", molmo_spaces.__file__)
print("MuJoCo:", mujoco.__version__)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.cuda.is_available())
print("Warp CUDA:", warp.is_cuda_available())
PY
```

### 5. List the fixed Franka PnP pools

```bash
python scripts/datagen/run_pipeline.py --list_pools
```

Each pool fixes the scene dataset, split, house, pickup object, and receptacle. A pool remains a research candidate until it passes the full behavior and artifact gates on the target machine.

### 6. Run a bounded Franka PnP datagen smoke

On native Linux with an NVIDIA GPU, select a GPU for Warp parallel IK. MuJoCo physics remains on CPU:

```bash
export CUDA_VISIBLE_DEVICES=0

python scripts/datagen/run_pipeline.py \
  --robot droid --policy planner --task_type pick_and_place \
  --pool molmodata_potato_bowl_1716 \
  --samples_per_house 1 --randomize_fixed_pickup_pose \
  --filter_for_successful_trajectories \
  --disable_action_noise --require_clean_success \
  --device gpu --num_workers 1 \
  --seed 111 --run_name_prefix fresh_clone_smoke
```

Here `--robot droid` means `FrankaRobotConfig` with `FrankaDroidCameraSystem`. Use `--device cpu` only as a slower diagnostic fallback. WSL2 does not expose the NVIDIA EGL device extension required by MolmoSpaces headless rendering, so complete GPU-rendered datagen acceptance requires native Linux with the NVIDIA EGL vendor configuration.

`samples_per_house=1` requests one saved trajectory; it does not prove that a valid demonstration was produced. Accept a datagen run only when the process exits successfully, a non-empty HDF5 trajectory and expected videos are present, arrays are finite, the task identity matches the pool, planner phases cover full Pick-and-Place behavior, replay/video shows approach through stable release, and `--require_clean_success` observed no planner retry.

Generated files are written under the MolmoSpaces resource datagen directory printed by the run. Config construction, scene loading, or an HDF5 file alone is not datagen success.

## Continue to MimicGen

Complete the Franka datagen and artifact gates before using its HDF5 as MimicGen source input.

### 1. Fetch pinned MimicGen and robomimic dependencies

`vendor/` is intentionally not committed to Git. Fetch the upstream repositories at the commits used by this workline:

```bash
bash tools/setup_mimicgen_dependency.sh
```

The script currently pins:

- MimicGen: `72bd767c255545f462e7ccfb2731f2e5d4c1d9bb`
- robomimic: `e10526b9a40c78b41f1e37e60041dc0ec0a5f60f`

Install them in editable mode:

```bash
pip install -e vendor/robomimic
pip install -e vendor/mimicgen
```

Set dependency roots and `PYTHONPATH`:

```bash
export MIMICGEN_ROOT=$PWD/vendor/mimicgen
export ROBOMIMIC_ROOT=$PWD/vendor/robomimic
export PYTHONPATH=$PWD:$MIMICGEN_ROOT:$ROBOMIMIC_ROOT:${PYTHONPATH:-}
```

If existing local checkouts are used instead, point `MIMICGEN_ROOT` and `ROBOMIMIC_ROOT` to those directories.

### 2. Set runtime paths

```bash
export MOLMOSPACES_ROOT=$PWD
export MOLMOSPACES_PYTHON=python
export MOLMOSPACES_PNP_WORKDIR=$PWD/runtime/mimicgen_pick_and_place
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export NLTK_DATA=${NLTK_DATA:-$HOME/nltk_data}
export MOLMOSPACES_NLTK_DATA=$NLTK_DATA

mkdir -p "$MOLMOSPACES_PNP_WORKDIR"/{artifacts/seeds,artifacts/mimicgen_pnp,data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards,logs}
```

`HF_HOME` is used by robomimic's CLIP language embedding utility. `NLTK_DATA` / `MOLMOSPACES_NLTK_DATA` are useful when MolmoSpaces needs a local WordNet cache.

### 3. Choose the source input

For newly generated Franka HDF5, point the selector at one run directory or a parent containing several runs:

```bash
PNP_SELECT_N=50 $MOLMOSPACES_PYTHON src/pnp/select_pnp_50_source_pool.py \
  --franka-datagen-root /path/to/datagen/pick_and_place_planner_v1
```

Franka mode requires terminal and persistent success, planner phases `0..9`, terminal `task_info.success=true`, required replay fields, and unique initial-state/action fingerprints. Its output is compatible with the downstream MimicGen scripts.

Alternatively, the original MolmoBot shard route remains supported. Download the official Pick-and-Place validation shard and place it here:

```text
runtime/mimicgen_pick_and_place/data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar
```

The repository includes lightweight manifests and summaries but not official data shards or generated artifacts.

### 4. First-run resource cache notes

MolmoSpaces may download/extract iTHOR assets on first use. If a run is interrupted during extraction, you may see an error like:

```text
Directory path exists on disk but is not recorded in the cache manifest
```

Do not hand-edit the manifest. Move the reported unregistered resource directory aside, then rerun so the resource manager can extract and register it cleanly. For example:

```bash
mkdir -p "$HOME/.cache/molmo-spaces-resources_broken_backup"
mv "$HOME/.cache/molmo-spaces-resources/objects/thor/20251117" \
  "$HOME/.cache/molmo-spaces-resources_broken_backup/thor_20251117_$(date +%Y%m%d_%H%M%S)"
```

If your network requires a proxy, export it before the first asset/model download.

### 5. Run a MolmoBot-source replay smoke check

Copy a manifest into the runtime work directory:

```bash
cp results/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json \
  "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/"
```

Inspect source candidates:

```bash
$MOLMOSPACES_PYTHON src/pnp/inspect_source_candidates.py
```

Replay one source trajectory to verify the replay pipeline works:

```bash
$MOLMOSPACES_PYTHON src/pnp/replay_source_episode.py --seed-index 0 --save-videos
```

## MolmoSpaces Workline Portfolio

The workline table above is mirrored in [`docs/worklines/README.md`](docs/worklines/README.md). Each workline has a canonical README with details, even when its raw HDF5/videos are excluded from Git.

MolmoAct2 official `sim_eval` success, MolmoSpaces adapter diagnostics, bimanual browser teleoperation, custom-scene YAM baseline, iTHOR source-demo infrastructure, and Pick-and-Place MimicGen rollouts are separate evidence layers and should not be conflated.

## Pick-and-Place Integration Workflow

The Pick-and-Place pipeline accepts either validated Franka datagen HDF5 or MolmoBot-Data source trajectories, then generates MimicGen rollouts in a reproducible sequence:

1. inspect or prepare source-candidate metadata;
2. replay source trajectories and collect MimicGen datagen information;
3. convert selected sources into a robomimic/MimicGen source HDF5;
4. parse the source HDF5 with MimicGen;
5. generate MolmoSpaces rollouts and save videos;
6. optionally collect accepted rollouts with action-hash deduplication.

### Collect MimicGen datagen information

Replay source trajectories to extract the observations and actions MimicGen needs for spatial transform:

```bash
$MOLMOSPACES_PYTHON src/pnp/collect_homogeneous_datagen_info.py \
  --seed-index 0 \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_homogeneous_foodlike_bowl_10candidate_v3"
```

### Convert sources to a robomimic/MimicGen source HDF5

Bundle selected source trajectories into a single robomimic-compatible HDF5:

```bash
$MOLMOSPACES_PYTHON src/pnp/convert_seed_set_to_robomimic.py \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5"
```

### Parse the source HDF5 with MimicGen

Load the source HDF5 into MimicGen's dataset format and inspect properties:

```bash
PNP_SOURCE_HDF5="$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5" \
$MOLMOSPACES_PYTHON src/pnp/parse_source_dataset.py
```

### Generate a MolmoSpaces rollout

Transform a source trajectory into a new scene layout using MimicGen's object-centric spatial transform:

```bash
$MOLMOSPACES_PYTHON src/pnp/generate_pick_place_rollout.py \
  --source-hdf5 "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5" \
  --target-manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --demo-keys demo_2 \
  --seed-index 2 \
  --out-name example_target02_src02 \
  --interp 1 --fixed 0 --noise 0.0 \
  --transform-first-robot-pose \
  --post-hold-steps 30 \
  --save-videos
```

### 50-demo cross-subtask MimicGen route

The workflow above uses one source demo per rollout. The cross-subtask route uses a broader 50-demo source pool and calls MimicGen with `select_src_per_subtask=True`, allowing different subtasks in a single rollout to be sampled from different source demos:

```bash
# Select the broad source pool from the official MolmoBot shard.
$MOLMOSPACES_PYTHON src/pnp/select_pnp_50_source_pool.py

# Collect strict replay + datagen_info for candidate sources.
bash src/pnp/run_collect_50cross_datagen_parallel.sh
# or for one candidate:
$MOLMOSPACES_PYTHON src/pnp/collect_datagen_info_50cross.py \
  --seed-index 0 \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_50demo_crossmix.json" \
  --out-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_50cross"

# Build the 50-demo MimicGen source HDF5 from hard-pass sources.
$MOLMOSPACES_PYTHON src/pnp/convert_seed_set_to_robomimic_50cross.py \
  --accepted all \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_50demo_crossmix.json" \
  --replay-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_50cross" \
  --out "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_50demo_crossmix_aligned.hdf5"

# Run a select-src-per-subtask pilot.
bash src/pnp/run_50cross_selectsrc_pilot.sh
```

Broad random mixing across many houses, objects, and receptacles can expose geometry, contact, IK, and subtask-stitching compatibility problems. This route is diagnostic; a compatibility-filtered cross-subtask route is the next step toward a scalable success-rate experiment.

### Batch collection

Collect successes with the non-deduplicated baseline collector (collects any rollout that meets the success criteria, may include duplicates):

```bash
TARGET_SUCCESS=100 MAX_ATTEMPTS=800 bash src/pnp/collect_uniform_successes.sh
```

Collect successes with action-hash deduplication (skips rollouts whose action sequence matches a previously collected trajectory):

```bash
PREVIOUS_COLLECTOR_RUN=logs/collect_uniform_successes_YYYYMMDD_HHMMSS \
TARGET_SUCCESS=100 MAX_ATTEMPTS=500 \
bash src/pnp/collect_unique_highyield_successes.sh
```

## Bimanual YAM Browser Teleoperation

Browser-based keyboard teleoperation for bimanual YAM scenes. The supported public entrypoint is the keyboard teleoperation bridge, which performs the current strict tabletop initialization before serving the browser UI. The older read-only viewer script is not the recommended reproduction entrypoint for this workline.

### Environment requirements

- **GPU rendering is required** for usable frame rates. CPU software rendering (OSMesa) with three cameras gives ~3 FPS, which is too slow for responsive keyboard control.
- **NVIDIA EGL headless rendering** is the tested configuration: a Linux host with an NVIDIA GPU, the NVIDIA proprietary driver, and EGL vendor files (`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`). The teleop automatically uses `MUJOCO_GL=egl` via MolmoSpaces.
- **WSL2 is not supported** for this workline. WSL2 uses Mesa EGL, which does not expose `EGL_EXT_platform_device` — the extension MolmoSpaces relies on for headless GPU rendering. If you are on WSL2, run the teleop on a remote Linux GPU server instead (see below).

### Running locally (Linux + NVIDIA GPU)

Run the teleoperation bridge:

```bash
HF_HOME=/mnt/vqa/.cache/huggingface \
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_keyboard_teleop.py \
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

If the CLIP model is cached elsewhere, set `HF_HOME` to that directory. If you see a SOCKS proxy error on first run, install `httpx[socks]` first (see step 3 above).

Open `http://127.0.0.1:8765` after the terminal prints the local teleoperation URL.

### Running on a remote GPU server

If you run the teleop on a remote Linux GPU server, set up an SSH tunnel to forward the browser port to your local machine:

```bash
# On your local machine:
ssh -L 8765:127.0.0.1:8765 user@your-gpu-server
```

Then open `http://127.0.0.1:8765` in your local browser.

### Browser controls

Click the page first; `Tab` switches active arm; `W/S/A/D` moves in the visual plane; `E/Q` raises/lowers; arrow keys control pitch/yaw; `Z/C` roll; `F` toggles the active gripper. If initialization fails, increase `--initialization-max-attempts` or inspect `runtime/bimanual_yam_initialization_report.json`.

## Included Results Snapshot

- **Heterogeneous Pick-and-Place whole-source generation**: `10/10` accepted generated rollouts with full rollout, final success, persistent success, and 30-step post-hold. See `results/whole_source_transformfirst_summary.json`.
- **Homogeneous foodlike-to-bowl pilot**: strict automatic success `13/100`; reviewed visual success `15/100` after two one-frame trace-glitch cases.
- **50-demo cross-subtask source pool**: `51` strict replay/datagen-info hard-pass candidates were screened, `50` were selected into `robomimic_pnp_50demo_crossmix_aligned.hdf5`, with `9286` total source action rows. See `results/pnp_50cross_selected_hardpass_indices.json` and `results/robomimic_pnp_50demo_crossmix_aligned.summary.json`.
- **Broad random `select_src_per_subtask=True` pilot**: first completed samples exposed geometry/contact/source-compatibility issues. Lightweight logs and generation traces are under `results/50cross_selectsrc_pilot_20260727_182533/`; large binary videos/HDF5/arrays are excluded from Git.
- **Uniform collector live snapshot**: non-deduplicated progress in `results/collector_uniform_summary_live.json`.
- **High-yield deduplicated collector live snapshot**: unique accepted progress in `results/collector_highyield_dedup_summary_live.json`.

<details>
<summary>Demo media</summary>

### Generated Rollout Examples

[![Generated rollout examples](media/gif/heterogeneous_generated_examples.gif)](media/heterogeneous_generated_examples.mp4)

[Open MP4](media/heterogeneous_generated_examples.mp4)

### Source Candidates

[![Foodlike source candidates](media/gif/foodlike_source_candidates.gif)](media/foodlike_source_candidates_grid.mp4)

[Open MP4](media/foodlike_source_candidates_grid.mp4)

### Pilot Outcomes

[![Foodlike pilot outcomes](media/gif/foodlike_pilot_outcomes.gif)](media/foodlike_pilot_outcomes.mp4)

[Open MP4](media/foodlike_pilot_outcomes.mp4)

</details>

## Repository Layout

```text
molmo_spaces/             Upstream MolmoSpaces Python package
scripts/                  Upstream MolmoSpaces scripts
configs/, examples/, docs/ Upstream configuration, examples, and documentation
src/pnp/                  Pick-and-Place MimicGen integration scripts
src/bimanual_yam/         Bimanual YAM diagnostics and browser keyboard teleoperation
results/                  Lightweight JSON manifests and result summaries
media/                    Small public demo videos for the GitHub README
docs/experiments.md       Public experiment notes and evidence boundaries
docs/upstream_molmospaces_readme.md  Original upstream MolmoSpaces README
tools/setup_mimicgen_dependency.sh   Helper to fetch MimicGen and robomimic into vendor/
```

## Evidence Boundaries

- Source trajectories are synthetic planner expert trajectories from MolmoBot-Data, not human demonstrations.
- Replay success and parser success are prerequisites, not generated-demo success.
- Accepted generated demonstrations require a real MolmoSpaces simulator rollout, `final_success=true`, post-hold stability, and saved artifacts.

## Attribution

- Upstream MolmoSpaces: Allen Institute for AI, Apache License 2.0. The upstream source and license are retained in this repository.
- Upstream MimicGen: NVIDIA / NVlabs. MimicGen was introduced by Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox. MimicGen code is released under the NVIDIA Source Code License, and MimicGen datasets are released under CC-BY 4.0.
- MolmoSpaces Integration and release organization: Kunyu Yang, Institute of Trustworthy Embodied Intelligence, Fudan University.
- Details: `AUTHORS.md`.

## License

The upstream MolmoSpaces code is distributed under the Apache License 2.0; see `LICENSE`. Third-party dependencies and datasets retain their own licenses. The integration code in this repository is provided as research code under the same repository license unless a file states otherwise.
