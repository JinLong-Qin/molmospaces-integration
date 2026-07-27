<div align="center">
  <img src="media/banner.svg" alt="MolmoSpaces x MimicGen" width="980" />
</div>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <strong>MolmoSpaces x MimicGen</strong> combines the upstream MolmoSpaces codebase with an integration layer for Pick-and-Place trajectory generation and bimanual YAM browser/keyboard teleoperation.
</p>

<p align="center">
  A new user can clone this repository, install MolmoSpaces, fetch MimicGen and robomimic, place the official MolmoBot-Data shard under <code>runtime/</code>, and run the included source-replay and generation scripts.
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

<p align="center"><em>Previews are intentionally compact; open the MP4 links for the full-size demos.</em></p>

## What is included

- Upstream MolmoSpaces source snapshot.
- Pick-and-Place MimicGen integration scripts under `src/pnp/`.
- Bimanual YAM browser visualization and keyboard teleoperation scripts under `src/bimanual_yam/`.
- Lightweight manifests and result summaries under `results/`.
- Small README demo media under `media/`.
- Public documentation and attribution files.

Large runtime data are intentionally excluded from Git: official shards, generated HDF5 files, rollout directories, simulator logs, PID files, caches, and internal planning ledgers.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/yanggoumao2/molmospaces-mimicgen.git
cd molmospaces-mimicgen
```

SSH also works:

```bash
git clone git@github.com:yanggoumao2/molmospaces-mimicgen.git
cd molmospaces-mimicgen
```

### 2. Create a Python environment

MolmoSpaces uses Python 3.11.

With conda:

```bash
conda create -n molmospaces-mimicgen python=3.11
conda activate molmospaces-mimicgen
python -m pip install --upgrade pip setuptools wheel
```

With `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
```

### 3. Install MolmoSpaces

For the MuJoCo-based Pick-and-Place integration:

```bash
pip install -e ".[mujoco]"
```

Optional upstream extras can be installed as needed, for example:

```bash
pip install -e ".[mujoco,grasp,housegen]"
```

See `docs/upstream_molmospaces_readme.md` for upstream MolmoSpaces installation details.

### 4. Fetch MimicGen and robomimic

The integration scripts use MimicGen and robomimic. Fetch both into `vendor/`:

```bash
bash tools/setup_mimicgen_dependency.sh
```

Install them in editable mode when their package metadata is available:

```bash
pip install -e vendor/robomimic
pip install -e vendor/mimicgen
```

Set dependency roots:

```bash
export MIMICGEN_ROOT=$PWD/vendor/mimicgen
export ROBOMIMIC_ROOT=$PWD/vendor/robomimic
```

If existing local checkouts are used instead, point `MIMICGEN_ROOT` and `ROBOMIMIC_ROOT` to those directories.

### 5. Set runtime paths

```bash
export MOLMOSPACES_ROOT=$PWD
export MOLMOSPACES_PYTHON=python
export MOLMOSPACES_PNP_WORKDIR=$PWD/runtime/mimicgen_pick_and_place

mkdir -p "$MOLMOSPACES_PNP_WORKDIR"/{artifacts/seeds,artifacts/mimicgen_pnp,data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards,logs}
```

Optional NLTK cache:

```bash
export MOLMOSPACES_NLTK_DATA=/path/to/nltk_data
```

### 6. Place the MolmoBot-Data shard

Download the official MolmoBot Pick-and-Place validation shard and place it here:

```text
runtime/mimicgen_pick_and_place/data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar
```

This repository includes lightweight manifests and summaries, but not official data shards or generated artifacts.

### 7. Run a smoke check

Copy a manifest into the runtime work directory:

```bash
cp results/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json \
  "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/"
```

Inspect source candidates:

```bash
$MOLMOSPACES_PYTHON src/pnp/inspect_source_candidates.py
```

Replay one source trajectory:

```bash
$MOLMOSPACES_PYTHON src/pnp/replay_source_episode.py --seed-index 0 --save-videos
```

A successful replay only verifies source-trajectory replay. It is not yet a generated MimicGen rollout.

## Pick-and-Place Integration Workflow

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

These snapshots are progress evidence, not a final 100-success dataset.

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
- Large generated artifacts are intentionally kept outside Git.

## Attribution

- Upstream MolmoSpaces: Allen Institute for AI, Apache License 2.0. The upstream source and license are retained in this repository.
- Upstream MimicGen: NVIDIA / NVlabs. MimicGen was introduced by Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox. MimicGen code is released under the NVIDIA Source Code License, and MimicGen datasets are released under CC-BY 4.0.
- MolmoSpaces x MimicGen integration and release organization: Kunyu Yang, Institute of Trustworthy Embodied Intelligence, Fudan University.
- Details: `AUTHORS.md`.

## License

The upstream MolmoSpaces code is distributed under the Apache License 2.0; see `LICENSE`. Third-party dependencies and datasets retain their own licenses. The MimicGen integration code in this repository is provided as research code under the same repository license unless a file states otherwise.
