<div align="center">
  <img src="media/banner.svg" alt="MolmoSpaces Integration" width="980" />
</div>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <strong>MolmoSpaces Integration</strong> — Pick-and-Place MimicGen 轨迹生成、bimanual YAM 遥操作及其他 MolmoSpaces 集成工作线。
</p>

<p align="center">
  <img src="media/gif/heterogeneous_generated_examples.gif" alt="Generated rollout examples" width="245" />
  &nbsp;
  <img src="media/gif/source_candidate_example.gif" alt="Source candidate example" width="245" />
  &nbsp;
  <img src="media/gif/foodlike_pilot_outcomes.gif" alt="Pilot outcomes" width="245" />
</p>

<p align="center">
  <a href="media/heterogeneous_generated_examples.mp4">生成示例</a>
  &nbsp;·&nbsp;
  <a href="media/foodlike_source_candidates_grid.mp4">源轨迹候选</a>
  &nbsp;·&nbsp;
  <a href="media/foodlike_pilot_outcomes.mp4">Pilot 结果</a>
</p>

## 工作线

本仓库包含多条集成工作线，每条在 `docs/worklines/` 下有独立的 README。

| 工作线 | README | 代码 | 结果 | 状态 |
|---|---|---|---|---|
| MimicGen Pick-and-Place | [`docs/worklines/mimicgen_pick_and_place/README.md`](docs/worklines/mimicgen_pick_and_place/README.md) | [`src/pnp/`](src/pnp/) | [`results/workline_index/mimicgen_pick_and_place.md`](results/workline_index/mimicgen_pick_and_place.md) | Active |
| 50-demo MimicGen cross-subtask | [`docs/worklines/mimicgen_50cross/README.md`](docs/worklines/mimicgen_50cross/README.md) | [`src/pnp/*50cross*`](src/pnp/) | [`results/50cross_selectsrc_pilot_20260727_182533/`](results/50cross_selectsrc_pilot_20260727_182533/) | Diagnostic |
| Bimanual YAM browser teleop | [`docs/worklines/bimanual_yam_browser_teleop/README.md`](docs/worklines/bimanual_yam_browser_teleop/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | Infrastructure |
| iTHOR bimanual YAM | [`docs/worklines/ithor_bimanual_yam/README.md`](docs/worklines/ithor_bimanual_yam/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | In progress |
| Custom-scene bimanual YAM baseline | [`docs/worklines/bimanual_yam_source_baseline/README.md`](docs/worklines/bimanual_yam_source_baseline/README.md) | — | [`results/workline_index/bimanual_yam_source_baseline.md`](results/workline_index/bimanual_yam_source_baseline.md) | Completed |
| MolmoAct2 → MolmoSpaces legacy | [`docs/worklines/molmoact2_legacy/README.md`](docs/worklines/molmoact2_legacy/README.md) | [`src/molmoact2_legacy/`](src/molmoact2_legacy/) | [`results/workline_index/molmoact2_integration_legacy.md`](results/workline_index/molmoact2_integration_legacy.md) | Legacy |
| MolmoSpaces bounded reproduction | [`docs/worklines/molmospaces_official_reproduction/README.md`](docs/worklines/molmospaces_official_reproduction/README.md) | — | [`results/workline_index/molmospaces_official_reproduction.md`](results/workline_index/molmospaces_official_reproduction.md) | Completed |
| MimicGen single-arm groundwork | [`docs/worklines/mimicgen_single_arm/README.md`](docs/worklines/mimicgen_single_arm/README.md) | 已合并到 [`src/pnp/`](src/pnp/) | [`results/workline_index/mimicgen_single_arm.md`](results/workline_index/mimicgen_single_arm.md) | Historical |

大体积运行数据（官方 shard、生成的 HDF5、rollout 目录、仿真日志、视频）不提交到 Git。

## Quick Start

Clone 仓库，安装 MolmoSpaces 和依赖，放置官方 MolmoBot-Data shard，运行 smoke check。

### 1. Clone

```bash
git clone https://github.com/yanggoumao2/molmospaces-integration.git
cd molmospaces-integration
```

### 2. 创建 Python 环境

MolmoSpaces 需要 Python 3.11。

```bash
conda create -n molmospaces-integration python=3.11
conda activate molmospaces-integration
python -m pip install --upgrade pip setuptools wheel
```

或使用 `uv`：

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
```

### 3. 安装 MolmoSpaces

```bash
pip install -e ".[mujoco]"
```

可选 extras（house generation、grasping）：

```bash
pip install -e ".[mujoco,grasp,housegen]"
```

上游安装细节见 `docs/upstream_molmospaces_readme.md`。

### 4. 拉取 MimicGen 和 robomimic

集成脚本依赖 MimicGen 和 robomimic，拉到 `vendor/`：

```bash
bash tools/setup_mimicgen_dependency.sh
pip install -e vendor/robomimic
pip install -e vendor/mimicgen
```

```bash
export MIMICGEN_ROOT=$PWD/vendor/mimicgen
export ROBOMIMIC_ROOT=$PWD/vendor/robomimic
```

### 5. 设置运行路径

```bash
export MOLMOSPACES_ROOT=$PWD
export MOLMOSPACES_PYTHON=python
export MOLMOSPACES_PNP_WORKDIR=$PWD/runtime/mimicgen_pick_and_place

mkdir -p "$MOLMOSPACES_PNP_WORKDIR"/{artifacts/seeds,artifacts/mimicgen_pnp,data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards,logs}
```

### 6. 放置 MolmoBot-Data shard

下载官方 MolmoBot Pick-and-Place validation shard，放到：

```text
runtime/mimicgen_pick_and_place/data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar
```

### 7. Smoke check

复制 seed manifest 并检查源轨迹候选：

```bash
cp results/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json \
  "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/"

$MOLMOSPACES_PYTHON src/pnp/inspect_source_candidates.py
```

回放一条源轨迹，验证回放管线：

```bash
$MOLMOSPACES_PYTHON src/pnp/replay_source_episode.py --seed-index 0 --save-videos
```

## Pick-and-Place 流程

Pick-and-Place 管线从 MolmoBot-Data 源轨迹生成 MimicGen rollout：收集 datagen 信息、转换为 robomimic HDF5、用 MimicGen 解析、通过空间变换在新场景布局中生成 rollout。

### 1. 收集 MimicGen datagen 信息

回放源轨迹并提取 MimicGen 所需的观测和动作：

```bash
$MOLMOSPACES_PYTHON src/pnp/collect_homogeneous_datagen_info.py \
  --seed-index 0 \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_homogeneous_foodlike_bowl_10candidate_v3"
```

### 2. 转换为 robomimic HDF5

将选中的源轨迹打包为 robomimic 兼容的 HDF5：

```bash
$MOLMOSPACES_PYTHON src/pnp/convert_seed_set_to_robomimic.py \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5"
```

### 3. 用 MimicGen 解析 source dataset

```bash
PNP_SOURCE_HDF5="$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5" \
$MOLMOSPACES_PYTHON src/pnp/parse_source_dataset.py
```

### 4. 生成 rollout

用 MimicGen 的空间变换将源轨迹迁移到新场景布局：

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

### 50-demo cross-subtask 路线

cross-subtask 路线使用更大的 50-demo 源池，调用 MimicGen `select_src_per_subtask=True`，允许同一 rollout 的不同子任务从不同源 demo 采样：

```bash
# 从官方 MolmoBot shard 选取源池。
$MOLMOSPACES_PYTHON src/pnp/select_pnp_50_source_pool.py

# 收集候选的严格回放 + datagen_info。
bash src/pnp/run_collect_50cross_datagen_parallel.sh

# 从 hard-pass 源构建 50-demo MimicGen source HDF5。
$MOLMOSPACES_PYTHON src/pnp/convert_seed_set_to_robomimic_50cross.py \
  --accepted all \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_50demo_crossmix.json" \
  --replay-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_50cross" \
  --out "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_50demo_crossmix_aligned.hdf5"

# 运行 select-src-per-subtask pilot。
bash src/pnp/run_50cross_selectsrc_pilot.sh
```

批量收集 accepted rollout：

```bash
# 非去重 collector。
TARGET_SUCCESS=100 MAX_ATTEMPTS=800 bash src/pnp/collect_uniform_successes.sh
```

```bash
# action-hash 去重 collector。
PREVIOUS_COLLECTOR_RUN=logs/collect_uniform_successes_YYYYMMDD_HHMMSS \
TARGET_SUCCESS=100 MAX_ATTEMPTS=500 \
bash src/pnp/collect_unique_highyield_successes.sh
```

## Bimanual YAM 浏览器遥操作

bimanual YAM 场景的浏览器遥操作工具——浏览器内查看器和键盘控制桥接。

只读浏览器视频流：

```bash
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_viewer.py --host 127.0.0.1 --port 8765
```

键盘遥操作：

```bash
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_keyboard_teleop.py --host 127.0.0.1 --port 8765
```

## 结果

- **Heterogeneous PnP whole-source generation**：`10/10` accepted generated rollout，均满足 full rollout、`final_success=true`、persistent success 和 30-step post-hold（`results/whole_source_transformfirst_summary.json`）。
- **Homogeneous foodlike-to-bowl pilot**：自动通过 `13/100`，人工复核 `15/100`，含两个 one-frame trace-glitch case。
- **50-demo cross-subtask source pool**：`50` hard-pass sources，`9286` total action rows（`robomimic_pnp_50demo_crossmix_aligned.hdf5`）。见 `results/pnp_50cross_selected_hardpass_indices.json`、`results/robomimic_pnp_50demo_crossmix_aligned.summary.json`。
- **`select_src_per_subtask=True` pilot**：跨 house/物体/容器的随机混合暴露了 geometry 和兼容性问题。日志见 `results/50cross_selectsrc_pilot_20260727_182533/`。
- **Uniform collector snapshot**：`results/collector_uniform_summary_live.json`。
- **High-yield dedup collector snapshot**：`results/collector_highyield_dedup_summary_live.json`。

## 仓库结构

```text
molmo_spaces/             上游 MolmoSpaces Python 包
scripts/                  上游 MolmoSpaces 脚本
configs/, examples/, docs/ 上游配置、示例和文档
src/pnp/                  Pick-and-Place MimicGen 集成脚本
src/bimanual_yam/         bimanual YAM 诊断与浏览器遥操作
results/                  轻量 JSON manifest 和结果摘要
media/                    README 演示 GIF 和视频
tools/setup_mimicgen_dependency.sh   拉取 MimicGen 和 robomimic
```

## 引用

- **上游 MolmoSpaces**：Allen Institute for AI，Apache License 2.0。
- **上游 MimicGen**：NVIDIA / NVlabs（Ajay Mandlekar 等），NVIDIA Source Code License；数据集 CC-BY 4.0。
- **MolmoSpaces Integration**：Kunyu Yang，复旦大学可信具身智能研究院。见 `AUTHORS.md`。

## License

上游 MolmoSpaces 代码：Apache License 2.0（`LICENSE`）。集成代码：同仓库 license。
