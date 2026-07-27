# MolmoSpaces x MimicGen

[English](README.md) | [中文](README_zh.md)

## Demo Results

### Generated Rollout Examples

[![Generated rollout examples](media/gif/heterogeneous_generated_examples.gif)](media/heterogeneous_generated_examples.mp4)

[Open MP4](media/heterogeneous_generated_examples.mp4)

### Source Candidates and Pilot Outcomes

| Source candidates | Pilot outcomes |
| --- | --- |
| [![Foodlike source candidates](media/gif/foodlike_source_candidates.gif)](media/foodlike_source_candidates.mp4) | [![Foodlike pilot outcomes](media/gif/foodlike_pilot_outcomes.gif)](media/foodlike_pilot_outcomes.mp4) |
| [Open MP4](media/foodlike_source_candidates.mp4) | [Open MP4](media/foodlike_pilot_outcomes.mp4) |

This repository is a fork-style research snapshot that combines the MolmoSpaces codebase with a MolmoSpaces x MimicGen integration layer for Pick-and-Place trajectory generation and bimanual YAM browser/keyboard teleoperation.

The goal is that a new user can clone this repository, install the MolmoSpaces environment, fetch the external MimicGen dependency, download the official MolmoBot-Data shard, and run the included integration scripts.

## Attribution

- Upstream MolmoSpaces: Allen Institute for AI, Apache License 2.0. The upstream source and license are retained in this repository.
- Upstream MimicGen: NVIDIA / NVlabs. MimicGen was introduced by Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox. MimicGen code is released under the NVIDIA Source Code License, and MimicGen datasets are released under CC-BY 4.0.
- MolmoSpaces x MimicGen integration and release organization: Kunyu Yang, Institute of Trustworthy Embodied Intelligence, Fudan University.
- Details: `AUTHORS.md`.

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
tools/setup_mimicgen_dependency.sh   Helper to fetch MimicGen into vendor/
```

The repository intentionally excludes local runtime state and large artifacts: `.venv`, `work/`, official data shards, generated HDF5 files, full rollout video directories, simulator logs, PID files, cache directories, and internal planning ledgers. The only tracked videos are the small public demos in `media/`.

## Clone

```bash
git clone git@github.com:yanggoumao2/molmospaces-mimicgen.git
cd molmospaces-mimicgen
```

HTTPS also works:

```bash
git clone https://github.com/yanggoumao2/molmospaces-mimicgen.git
cd molmospaces-mimicgen
```

## Install MolmoSpaces

Use Python 3.11. The upstream package is installable from this repository root.

With conda:

```bash
conda create -n molmospaces-mimicgen python=3.11
conda activate molmospaces-mimicgen
pip install -e ".[mujoco]"
```

With `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[mujoco]"
```

Optional extras follow upstream MolmoSpaces conventions, for example `.[mujoco,grasp,housegen]`. See `docs/upstream_molmospaces_readme.md` for upstream installation details.

## Fetch MimicGen

The integration scripts need MimicGen and robomimic code. Fetch MimicGen into `vendor/mimicgen`:

```bash
bash tools/setup_mimicgen_dependency.sh
export MIMICGEN_ROOT=$PWD/vendor/mimicgen
```

If you already have a local MimicGen checkout, point `MIMICGEN_ROOT` to it instead.

## Environment Variables

Set these before running the integration scripts:

```bash
export MOLMOSPACES_ROOT=$PWD
export MOLMOSPACES_PYTHON=${MOLMOSPACES_PYTHON:-python}
export MIMICGEN_ROOT=${MIMICGEN_ROOT:-$PWD/vendor/mimicgen}
# optional, if using a local NLTK data cache
export MOLMOSPACES_NLTK_DATA=/path/to/nltk_data
```

Create the runtime work directory:

```bash
export MOLMOSPACES_PNP_WORKDIR=$PWD/runtime/mimicgen_pick_and_place
mkdir -p "$MOLMOSPACES_PNP_WORKDIR"/{artifacts/seeds,artifacts/mimicgen_pnp,data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards,logs}
```

## Data

Download the official MolmoBot Pick-and-Place validation shard into:

```text
runtime/mimicgen_pick_and_place/data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar
```

The repository includes lightweight manifests and summaries in `results/`, but not the official shard, generated HDF5 files, generated rollouts, or videos.

Primary included manifests:

```text
results/pnp_seed_manifest.json
results/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json
```

Copy a manifest into the runtime artifact directory when needed:

```bash
cp results/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json \
  "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/"
```

## Pick-and-Place Integration

Inspect candidate source trajectories:

```bash
$MOLMOSPACES_PYTHON src/pnp/inspect_source_candidates.py
```

Build a homogeneous foodlike-to-bowl manifest:

```bash
$MOLMOSPACES_PYTHON src/pnp/make_homogeneous_manifest.py
```

Replay a source trajectory:

```bash
$MOLMOSPACES_PYTHON src/pnp/replay_source_episode.py --seed-index 0 --save-videos
```

Collect MimicGen datagen information:

```bash
$MOLMOSPACES_PYTHON src/pnp/collect_homogeneous_datagen_info.py \
  --seed-index 0 \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_homogeneous_foodlike_bowl_10candidate_v3"
```

Convert selected source trajectories into a robomimic/MimicGen source HDF5:

```bash
$MOLMOSPACES_PYTHON src/pnp/convert_seed_set_to_robomimic.py \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5"
```

Parse the source HDF5 with MimicGen:

```bash
PNP_SOURCE_HDF5="$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5" \
$MOLMOSPACES_PYTHON src/pnp/parse_source_dataset.py
```

Generate one rollout:

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

Collect successes with the non-deduplicated baseline collector:

```bash
TARGET_SUCCESS=100 MAX_ATTEMPTS=800 bash src/pnp/collect_uniform_successes.sh
```

Collect successes with action-hash deduplication:

```bash
PREVIOUS_COLLECTOR_RUN=logs/collect_uniform_successes_YYYYMMDD_HHMMSS \
TARGET_SUCCESS=100 MAX_ATTEMPTS=500 \
bash src/pnp/collect_unique_highyield_successes.sh
```

## Bimanual YAM Browser Teleoperation

Read-only browser stream:

```bash
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_viewer.py --host 127.0.0.1 --port 8765
```

Keyboard teleoperation:

```bash
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_keyboard_teleop.py --host 127.0.0.1 --port 8765
```

The browser teleoperation path is a control and observation bridge. It is not by itself evidence of a completed task demonstration.

## Included Results Snapshot

- Heterogeneous Pick-and-Place whole-source generation: `10/10` accepted generated rollouts with full rollout, final success, persistent success, and 30-step post-hold. See `results/whole_source_transformfirst_summary.json`.
- Homogeneous foodlike-to-bowl pilot: strict automatic success `13/100`; reviewed visual success `15/100` after two one-frame trace-glitch cases.
- Uniform collector live snapshot: non-deduplicated progress in `results/collector_uniform_summary_live.json`.
- High-yield deduplicated collector live snapshot: unique accepted progress in `results/collector_highyield_dedup_summary_live.json`.

Do not claim final 100-success completion from these snapshots alone.

## Evidence Boundaries

- Source trajectories are synthetic planner expert trajectories from MolmoBot-Data, not human demonstrations.
- Replay success and parser success are prerequisites, not generated-demo success.
- Accepted generated demonstrations require a real MolmoSpaces simulator rollout, `final_success=true`, post-hold stability, and saved artifacts.
- Large generated artifacts are intentionally kept outside Git.

## License

The upstream MolmoSpaces code is distributed under the Apache License 2.0; see `LICENSE`. Third-party dependencies and datasets retain their own licenses. The MimicGen integration code in this repository is provided as research code under the same repository license unless a file states otherwise.
