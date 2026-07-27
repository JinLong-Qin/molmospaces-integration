# MolmoSpaces x MimicGen

A compact release of our MolmoSpaces x MimicGen integration work. The repository contains runnable integration scripts, browser keyboard teleoperation utilities for bimanual YAM, and lightweight result manifests from the Pick-and-Place experiments.

This is a platform-integration project: it shows how to use MolmoSpaces as the simulator backend for MimicGen-style trajectory generation. The source trajectories used here come from MolmoBot-Data synthetic planner expert rollouts. They must not be described as human demonstrations.

## What Is Included

- `src/pnp/`: Pick-and-Place source selection, exact replay, datagen-info extraction, robomimic/MimicGen HDF5 conversion, rollout generation, and success collectors.
- `src/bimanual_yam/`: bimanual YAM scene checks, browser camera streaming, keyboard teleoperation, tabletop initialization validation, and scripted source-demo diagnostics.
- `results/`: small JSON manifests and summaries from completed runs.
- `docs/`: public notes describing the experiment flow, evidence boundaries, and current status.

Large artifacts are intentionally not tracked by Git: official MolmoBot-Data shards, generated HDF5 files, videos, simulator logs, and rollout directories. Generate or download them using the commands below.

## Upstream Dependencies

Clone the upstream projects separately:

```bash
git clone https://github.com/allenai/molmospaces.git
git clone https://github.com/NVlabs/mimicgen.git
```

The scripts assume the MolmoSpaces checkout contains a work directory at:

```text
${MOLMOSPACES_ROOT}/runtime/mimicgen_pick_and_place
```

Set the environment variables before running:

```bash
export MOLMOSPACES_ROOT=/path/to/molmospaces
export MIMICGEN_ROOT=/path/to/mimicgen
export MOLMOSPACES_PNP_WORKDIR=${MOLMOSPACES_ROOT}/runtime/mimicgen_pick_and_place
export MOLMOSPACES_PYTHON=${MOLMOSPACES_ROOT}/.venv/bin/python
# optional, if your MolmoSpaces setup uses a local NLTK data cache
export MOLMOSPACES_NLTK_DATA=/path/to/nltk_data
```

Create the work layout:

```bash
mkdir -p ${MOLMOSPACES_PNP_WORKDIR}/{artifacts/seeds,artifacts/mimicgen_pnp,data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards,logs}
```

Copy this repository's `src/` directory into the MolmoSpaces checkout root, or run commands from `${MOLMOSPACES_ROOT}` with this repository on `PYTHONPATH`.

## Data

Download the official MolmoBot Pick-and-Place validation shard into:

```text
${MOLMOSPACES_PNP_WORKDIR}/data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar
```

The retained manifests in `results/` document the seeds used in our runs. The primary homogeneous source set is:

```text
results/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json
```

## Pick-and-Place Pipeline

Inspect candidate source trajectories:

```bash
${MOLMOSPACES_PYTHON} src/pnp/inspect_source_candidates.py
```

Build a homogeneous foodlike-to-bowl manifest:

```bash
${MOLMOSPACES_PYTHON} src/pnp/make_homogeneous_manifest.py
```

Replay selected source trajectories and collect MimicGen datagen information:

```bash
${MOLMOSPACES_PYTHON} src/pnp/replay_source_episode.py --seed-index 0 --save-videos
${MOLMOSPACES_PYTHON} src/pnp/collect_homogeneous_datagen_info.py --seed-index 0 \
  --manifest artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json \
  --out-root artifacts/replay_pnp_exact_homogeneous_foodlike_bowl_10candidate_v3
```

Convert selected seeds into robomimic/MimicGen source HDF5:

```bash
${MOLMOSPACES_PYTHON} src/pnp/convert_seed_set_to_robomimic.py \
  --manifest artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json \
  --out artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5
```

Parse the source dataset with MimicGen:

```bash
PNP_SOURCE_HDF5=artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5 \
${MOLMOSPACES_PYTHON} src/pnp/parse_source_dataset.py
```

Generate one Pick-and-Place rollout:

```bash
${MOLMOSPACES_PYTHON} src/pnp/generate_pick_place_rollout.py \
  --source-hdf5 artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5 \
  --target-manifest artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json \
  --demo-keys demo_2 \
  --seed-index 2 \
  --out-name example_target02_src02 \
  --interp 1 --fixed 0 --noise 0.0 \
  --transform-first-robot-pose \
  --post-hold-steps 30 \
  --save-videos
```

Collect successes using the original uniform schedule:

```bash
TARGET_SUCCESS=100 MAX_ATTEMPTS=800 bash src/pnp/collect_uniform_successes.sh
```

Collect action-deduplicated high-yield successes:

```bash
PREVIOUS_COLLECTOR_RUN=logs/collect_uniform_successes_YYYYMMDD_HHMMSS \
TARGET_SUCCESS=100 MAX_ATTEMPTS=500 \
bash src/pnp/collect_unique_highyield_successes.sh
```

The high-yield collector hashes `generated_actions.npy` and counts only unique successful trajectories.

## Bimanual YAM Browser Teleoperation

These scripts are a separate bimanual YAM workline. They provide a browser bridge and keyboard controller for MolmoSpaces bimanual YAM experiments.

Read-only browser stream:

```bash
${MOLMOSPACES_PYTHON} src/bimanual_yam/browser_viewer.py --host 127.0.0.1 --port 8765
```

Keyboard teleoperation:

```bash
${MOLMOSPACES_PYTHON} src/bimanual_yam/browser_keyboard_teleop.py --host 127.0.0.1 --port 8765
```

The browser page supports active-arm switching, Cartesian keyboard motion, gripper toggle, stale-input hold, invalid-input rejection, and loopback-only serving. It is a teleoperation/control bridge, not by itself a successful task demonstration.

## Current Results Snapshot

The retained summaries document these checkpoints:

- Heterogeneous Pick-and-Place whole-source generation: `10/10` accepted generated rollouts with full rollout, `final_success=true`, persistent success, and 30-step post-hold. See `results/whole_source_transformfirst_summary.json`.
- Homogeneous foodlike-to-bowl pilot: strict automatic success `13/100`; reviewed visual success `15/100` after two one-frame trace-glitch cases.
- Uniform collector live snapshot: nominal `27/100` accepted, but not action-deduplicated. See `results/collector_uniform_summary_live.json`.
- High-yield deduplicated collector live snapshot: `19/100` unique accepted trajectories at the last included snapshot. See `results/collector_highyield_dedup_summary_live.json`.

The 100-success collection was still in progress at the time of this snapshot. Do not claim final 100-success completion from this repository alone.

## Evidence Boundaries

- Source trajectories in this release are synthetic planner expert trajectories from MolmoBot-Data, not human demonstrations.
- Replay success and MimicGen parser success are prerequisites, not generated-demo success.
- Accepted generated demos require a real MolmoSpaces simulator rollout, `final_success=true`, post-hold stability, and real saved artifacts.
- Video files are not tracked in Git; produce them with `--save-videos` or attach them through a release asset / external artifact store.

## License

This repository contains integration glue and experiment scripts. Upstream MolmoSpaces, MimicGen, robomimic, and MolmoBot-Data retain their own licenses and terms. Check those upstream licenses before redistributing data or derived artifacts.
