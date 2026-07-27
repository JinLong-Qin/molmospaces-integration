# MolmoSpaces Integration Workline Portfolio

This repository is a public MolmoSpaces integration portfolio. Each workline has the same documentation entrypoint:

- a canonical workline README under `docs/worklines/<workline>/README.md`;
- code entrypoints under `src/` when reusable scripts are committed;
- lightweight evidence under `results/` when it is small and public-safe;
- an inventory under `results/workline_index/` when the full runtime artifacts are too large for normal Git.

The worklines are intentionally documented at the same level. A workline without committed raw HDF5/video data is not less important; it means the raw artifact is an external runtime artifact and the repository keeps the reproducible scripts, summaries, and inventory instead.

## Workline map

| Workline | Canonical README | Code entrypoint | Lightweight evidence | Status | Evidence boundary |
|---|---|---|---|---|---|
| MimicGen Pick-and-Place | [`mimicgen_pick_and_place/README.md`](mimicgen_pick_and_place/README.md) | [`src/pnp/`](../../src/pnp/) | `results/*pnp*`, [`results/workline_index/mimicgen_pick_and_place.md`](../../results/workline_index/mimicgen_pick_and_place.md) | Active / primary | Real MolmoSpaces rollout scripts and summaries; large HDF5/videos are runtime artifacts. |
| 50-demo MimicGen cross-subtask route | [`mimicgen_50cross/README.md`](mimicgen_50cross/README.md) | [`src/pnp/*50cross*`](../../src/pnp/) | [`results/50cross_*`](../../results/) | Diagnostic | Closer to MimicGen cross-demo subtask recombination; broad random mixing did not yield accepted stable rollouts. |
| Bimanual YAM browser teleoperation | [`bimanual_yam_browser_teleop/README.md`](bimanual_yam_browser_teleop/README.md) | [`src/bimanual_yam/`](../../src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](../../results/workline_index/ithor_bimanual_yam.md) | Infrastructure | Browser visualization and keyboard control bridge; not by itself a successful task demonstration. |
| iTHOR bimanual YAM | [`ithor_bimanual_yam/README.md`](ithor_bimanual_yam/README.md) | [`src/bimanual_yam/`](../../src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](../../results/workline_index/ithor_bimanual_yam.md) | In progress / research workline | Official iTHOR-style scene work; formal source-demo success must be separate from scene/control checks. |
| Completed custom-scene bimanual YAM baseline | [`bimanual_yam_source_baseline/README.md`](bimanual_yam_source_baseline/README.md) | Inventory only in this repo | [`results/workline_index/bimanual_yam_source_baseline.md`](../../results/workline_index/bimanual_yam_source_baseline.md) | Completed evidence package | Custom tabletop/box baseline; not an official iTHOR result. |
| MolmoAct2 → MolmoSpaces legacy | [`molmoact2_legacy/README.md`](molmoact2_legacy/README.md) | [`src/molmoact2_legacy/`](../../src/molmoact2_legacy/) | [`results/workline_index/molmoact2_integration_legacy.md`](../../results/workline_index/molmoact2_integration_legacy.md) | Legacy / ended | Adapter/API and partial behavior diagnostics; stable MolmoSpaces task success was not established. |
| Bounded official MolmoSpaces reproduction | [`molmospaces_official_reproduction/README.md`](molmospaces_official_reproduction/README.md) | Inventory only in this repo | [`results/workline_index/molmospaces_official_reproduction.md`](../../results/workline_index/molmospaces_official_reproduction.md) | Completed bounded reproduction | Bounded entrypoint/assets evidence; not a complete benchmark reproduction. |
| MimicGen single-arm groundwork | [`mimicgen_single_arm/README.md`](mimicgen_single_arm/README.md) | Superseded by `src/pnp/` public scripts | [`results/workline_index/mimicgen_single_arm.md`](../../results/workline_index/mimicgen_single_arm.md) | Historical groundwork | Single-arm replay/alignment groundwork; not the current formal PnP release result. |

## Standard clone setup

Most runnable worklines start from the same repository setup:

```bash
git clone https://github.com/yanggoumao2/molmospaces-mimicgen.git
cd molmospaces-mimicgen
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[mujoco]"
./tools/setup_mimicgen_dependency.sh
export MOLMOSPACES_ROOT="$PWD"
export MOLMOSPACES_PNP_WORKDIR="$PWD/runtime/pnp"
export MIMICGEN_ROOT="$PWD/third_party/mimicgen"
export ROBOMIMIC_ROOT="$PWD/third_party/robomimic"
```

Place external official data under `runtime/` as described in the top-level README. The repository does not commit official data shards, generated HDF5/H5 files, raw NPY arrays, videos, simulator caches, PID/lock files, or private runtime logs.

## How to read evidence

A command that runs, a file that exists, a generated HDF5, and a visually successful rollout are different evidence levels. Each workline README states what can and cannot be claimed from its artifacts.
