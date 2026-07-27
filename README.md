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

## Worklines

| Workline | README | Code | Results | Status |
|---|---|---|---|---|
| MimicGen Pick-and-Place | [`docs/worklines/mimicgen_pick_and_place/README.md`](docs/worklines/mimicgen_pick_and_place/README.md) | [`src/pnp/`](src/pnp/) | [`results/workline_index/mimicgen_pick_and_place.md`](results/workline_index/mimicgen_pick_and_place.md) | Active |
| 50-demo MimicGen cross-subtask | [`docs/worklines/mimicgen_50cross/README.md`](docs/worklines/mimicgen_50cross/README.md) | [`src/pnp/*50cross*`](src/pnp/) | [`results/50cross_selectsrc_pilot_20260727_182533/`](results/50cross_selectsrc_pilot_20260727_182533/) | Diagnostic |
| Bimanual YAM browser teleop | [`docs/worklines/bimanual_yam_browser_teleop/README.md`](docs/worklines/bimanual_yam_browser_teleop/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | Infrastructure |
| iTHOR bimanual YAM | [`docs/worklines/ithor_bimanual_yam/README.md`](docs/worklines/ithor_bimanual_yam/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | In progress |
| Custom-scene bimanual YAM baseline | [`docs/worklines/bimanual_yam_source_baseline/README.md`](docs/worklines/bimanual_yam_source_baseline/README.md) | — | [`results/workline_index/bimanual_yam_source_baseline.md`](results/workline_index/bimanual_yam_source_baseline.md) | Completed |
| MolmoAct2 → MolmoSpaces legacy | [`docs/worklines/molmoact2_legacy/README.md`](docs/worklines/molmoact2_legacy/README.md) | [`src/molmoact2_legacy/`](src/molmoact2_legacy/) | [`results/workline_index/molmoact2_integration_legacy.md`](results/workline_index/molmoact2_integration_legacy.md) | Legacy |
| MolmoSpaces bounded reproduction | [`docs/worklines/molmospaces_official_reproduction/README.md`](docs/worklines/molmospaces_official_reproduction/README.md) | — | [`results/workline_index/molmospaces_official_reproduction.md`](results/workline_index/molmospaces_official_reproduction.md) | Completed |
| MimicGen single-arm groundwork | [`docs/worklines/mimicgen_single_arm/README.md`](docs/worklines/mimicgen_single_arm/README.md) | Superseded by [`src/pnp/`](src/pnp/) | [`results/workline_index/mimicgen_single_arm.md`](results/workline_index/mimicgen_single_arm.md) | Historical |

Large runtime data (official shards, generated HDF5, rollout directories, logs, videos) are excluded from Git. See each workline README for details.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/yanggoumao2/molmospaces-integration.git
cd molmospaces-integration
```

### 2. Create a Python environment

```bash
conda create -n molmospaces-integration python=3.11
conda activate molmospaces-integration
python -m pip install --upgrade pip setuptools wheel
```

Or with `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
```

### 3. Install

```bash
pip install -e ".[mujoco]"
```

For additional extras:

```bash
pip install -e ".[mujoco,grasp,housegen]"
```

See `docs/upstream_molmospaces_readme.md` for upstream installation details.

### 4. Fetch MimicGen and robomimic

```bash
bash tools/setup_mimicgen_dependency.sh
pip install -e vendor/robomimic
pip install -e vendor/mimicgen
```

```bash
export MIMICGEN_ROOT=$PWD/vendor/mimicgen
export ROBOMIMIC_ROOT=$PWD/vendor/robomimic
```

### 5. Set runtime paths

```bash
export MOLMOSPACES_ROOT=$PWD
export MOLMOSPACES_PYTHON=python
export MOLMOSPACES_PNP_WORKDIR=$PWD/runtime/mimicgen_pick_and_place

mkdir -p "$MOLMOSPACES_PNP_WORKDIR"/{artifacts/seeds,artifacts/mimicgen_pnp,data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards,logs}
```

### 6. Place the MolmoBot-Data shard

```text
runtime/mimicgen_pick_and_place/data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar
```

### 7. Smoke check

```bash
cp results/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json \
  "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/"

$MOLMOSPACES_PYTHON src/pnp/inspect_source_candidates.py
$MOLMOSPACES_PYTHON src/pnp/replay_source_episode.py --seed-index 0 --save-videos
```

## Pick-and-Place Workflow

1. Collect MimicGen datagen information:

```bash
$MOLMOSPACES_PYTHON src/pnp/collect_homogeneous_datagen_info.py \
  --seed-index 0 \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_homogeneous_foodlike_bowl_10candidate_v3"
```

2. Convert sources to robomimic HDF5:

```bash
$MOLMOSPACES_PYTHON src/pnp/convert_seed_set_to_robomimic.py \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5"
```

3. Parse source dataset with MimicGen:

```bash
PNP_SOURCE_HDF5="$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5" \
$MOLMOSPACES_PYTHON src/pnp/parse_source_dataset.py
```

4. Generate a rollout:

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

### 50-demo cross-subtask route

```bash
$MOLMOSPACES_PYTHON src/pnp/select_pnp_50_source_pool.py

bash src/pnp/run_collect_50cross_datagen_parallel.sh

$MOLMOSPACES_PYTHON src/pnp/convert_seed_set_to_robomimic_50cross.py \
  --accepted all \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_50demo_crossmix.json" \
  --replay-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_50cross" \
  --out "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_50demo_crossmix_aligned.hdf5"

bash src/pnp/run_50cross_selectsrc_pilot.sh
```

Collectors:

```bash
TARGET_SUCCESS=100 MAX_ATTEMPTS=800 bash src/pnp/collect_uniform_successes.sh
```

```bash
PREVIOUS_COLLECTOR_RUN=logs/collect_uniform_successes_YYYYMMDD_HHMMSS \
TARGET_SUCCESS=100 MAX_ATTEMPTS=500 \
bash src/pnp/collect_unique_highyield_successes.sh
```

## Bimanual YAM Browser Teleoperation

```bash
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_viewer.py --host 127.0.0.1 --port 8765
```

```bash
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_keyboard_teleop.py --host 127.0.0.1 --port 8765
```

## Results

- Heterogeneous PnP whole-source generation: `10/10` accepted (`results/whole_source_transformfirst_summary.json`).
- Homogeneous foodlike-to-bowl pilot: `13/100` auto-accepted, `15/100` visual-reviewed.
- 50-demo cross-subtask source pool: `50` hard-pass sources in `robomimic_pnp_50demo_crossmix_aligned.hdf5` (`9286` action rows). See `results/pnp_50cross_selected_hardpass_indices.json`, `results/robomimic_pnp_50demo_crossmix_aligned.summary.json`.
- `select_src_per_subtask=True` pilot: broad random mixing exposed geometry/compatibility issues. Logs under `results/50cross_selectsrc_pilot_20260727_182533/`.
- Uniform collector snapshot: `results/collector_uniform_summary_live.json`.
- High-yield dedup collector snapshot: `results/collector_highyield_dedup_summary_live.json`.

## Repository Layout

```text
molmo_spaces/             Upstream MolmoSpaces Python package
scripts/                  Upstream MolmoSpaces scripts
configs/, examples/, docs/ Upstream configuration, examples, and documentation
src/pnp/                  Pick-and-Place MimicGen integration scripts
src/bimanual_yam/         Bimanual YAM diagnostics and browser teleop
results/                  Lightweight JSON manifests and result summaries
media/                    Demo GIFs and videos for the README
tools/setup_mimicgen_dependency.sh   Fetch MimicGen and robomimic into vendor/
```

## Attribution

- Upstream MolmoSpaces: Allen Institute for AI, Apache License 2.0.
- Upstream MimicGen: NVIDIA / NVlabs (Ajay Mandlekar et al.), NVIDIA Source Code License; datasets under CC-BY 4.0.
- MolmoSpaces Integration: Kunyu Yang, Institute of Trustworthy Embodied Intelligence, Fudan University. See `AUTHORS.md`.

## License

Upstream MolmoSpaces code: Apache License 2.0 (`LICENSE`). Integration code: same repository license.
