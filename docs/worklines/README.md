# Worklines

| Workline | README | Code | Results | Status |
|---|---|---|---|---|
| MimicGen Pick-and-Place | [`mimicgen_pick_and_place/README.md`](mimicgen_pick_and_place/README.md) | [`src/pnp/`](../../src/pnp/) | [`results/workline_index/mimicgen_pick_and_place.md`](../../results/workline_index/mimicgen_pick_and_place.md) | Active |
| 50-demo MimicGen cross-subtask | [archive README](../../archive/docs/worklines/mimicgen_50cross/README.md) | [archive code](../../archive/pnp/legacy_50cross/) | [results/50cross_selectsrc_pilot_20260727_182533/](../../results/50cross_selectsrc_pilot_20260727_182533/) | Archived diagnostic |
| Bimanual YAM browser teleop | [`bimanual_yam_browser_teleop/README.md`](bimanual_yam_browser_teleop/README.md) | [`src/bimanual_yam/`](../../src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](../../results/workline_index/ithor_bimanual_yam.md) | Infrastructure |
| iTHOR bimanual YAM | [`ithor_bimanual_yam/README.md`](ithor_bimanual_yam/README.md) | [`src/bimanual_yam/`](../../src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](../../results/workline_index/ithor_bimanual_yam.md) | In progress |
| Custom-scene bimanual YAM baseline | [`bimanual_yam_source_baseline/README.md`](bimanual_yam_source_baseline/README.md) | — | [`results/workline_index/bimanual_yam_source_baseline.md`](../../results/workline_index/bimanual_yam_source_baseline.md) | Completed |
| MolmoAct2 → MolmoSpaces legacy | [`molmoact2_legacy/README.md`](molmoact2_legacy/README.md) | [`src/molmoact2_legacy/`](../../src/molmoact2_legacy/) | [`results/workline_index/molmoact2_integration_legacy.md`](../../results/workline_index/molmoact2_integration_legacy.md) | Legacy |
| MolmoSpaces bounded reproduction | [`molmospaces_official_reproduction/README.md`](molmospaces_official_reproduction/README.md) | — | [`results/workline_index/molmospaces_official_reproduction.md`](../../results/workline_index/molmospaces_official_reproduction.md) | Completed |
| MimicGen single-arm groundwork | [`mimicgen_single_arm/README.md`](mimicgen_single_arm/README.md) | Superseded by [`src/pnp/`](../../src/pnp/) | [`results/workline_index/mimicgen_single_arm.md`](../../results/workline_index/mimicgen_single_arm.md) | Historical |

Large runtime data (HDF5, videos, logs) are excluded from Git.

## Reproduction Boundary

This portfolio includes multiple worklines that do not share the same runtime requirements.

- The public `src/pnp/` MimicGen Pick-and-Place path uses MolmoBot source trajectories and the integration scripts in this repository.
- The official RB-Y1 scripted/planner datagen path uses `python -m molmo_spaces.data_generation.main ...` with configs such as `RBY1PickDataGenConfig` and `RBY1PickAndPlaceDataGenConfig`; this is distinct from the public `src/pnp/` integration scripts.
- Bimanual browser teleop is a separate infrastructure path again.

A fresh `pip install -e ".[mujoco]"` setup is enough for the public `src/pnp/` worklines, but it is not enough for the RB-Y1 scripted/planner datagen configs. Those require `curobo` and should follow the dedicated CuRobo notes in the top-level [`README.md`](../../README.md).

## Setup

```bash
git clone https://github.com/yanggoumao2/molmospaces-integration.git
cd molmospaces-integration
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[mujoco]"
./tools/setup_mimicgen_dependency.sh
export MOLMOSPACES_ROOT="$PWD"
export MOLMOSPACES_PNP_WORKDIR="$PWD/runtime/pnp"
export MIMICGEN_ROOT="$PWD/vendor/mimicgen"
export ROBOMIMIC_ROOT="$PWD/vendor/robomimic"
```

If you specifically need official RB-Y1 scripted/planner expert generation, install and verify CuRobo before trying `RBY1PickDataGenConfig` or `RBY1PickAndPlaceDataGenConfig`, then verify planner-server connectivity for your own environment.
