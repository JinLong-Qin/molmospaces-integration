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

<p align="center"><em>预览保持紧凑；点击 MP4 链接查看完整尺寸演示。</em></p>

## 包含内容

本仓库是 MolmoSpaces 集成工作线合集。首页直接列出全部公开 workline，读者无需翻子目录即可了解仓库包含的所有方向。

| 工作线 | 详细 README | 代码入口 | 证据 / inventory | 状态 |
|---|---|---|---|---|
| MimicGen Pick-and-Place | [`docs/worklines/mimicgen_pick_and_place/README.md`](docs/worklines/mimicgen_pick_and_place/README.md) | [`src/pnp/`](src/pnp/) | [`results/workline_index/mimicgen_pick_and_place.md`](results/workline_index/mimicgen_pick_and_place.md) | Active / primary |
| 50-demo MimicGen cross-subtask route | [`docs/worklines/mimicgen_50cross/README.md`](docs/worklines/mimicgen_50cross/README.md) | [`src/pnp/*50cross*`](src/pnp/) | [`results/50cross_selectsrc_pilot_20260727_182533/`](results/50cross_selectsrc_pilot_20260727_182533/) | Diagnostic |
| Bimanual YAM browser teleop | [`docs/worklines/bimanual_yam_browser_teleop/README.md`](docs/worklines/bimanual_yam_browser_teleop/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | Infrastructure |
| iTHOR bimanual YAM | [`docs/worklines/ithor_bimanual_yam/README.md`](docs/worklines/ithor_bimanual_yam/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | In progress |
| Custom-scene bimanual YAM baseline | [`docs/worklines/bimanual_yam_source_baseline/README.md`](docs/worklines/bimanual_yam_source_baseline/README.md) | Inventory / regeneration entrypoints | [`results/workline_index/bimanual_yam_source_baseline.md`](results/workline_index/bimanual_yam_source_baseline.md) | Completed |
| MolmoAct2 → MolmoSpaces legacy | [`docs/worklines/molmoact2_legacy/README.md`](docs/worklines/molmoact2_legacy/README.md) | [`src/molmoact2_legacy/`](src/molmoact2_legacy/) | [`results/workline_index/molmoact2_integration_legacy.md`](results/workline_index/molmoact2_integration_legacy.md) | Legacy |
| MolmoSpaces bounded reproduction | [`docs/worklines/molmospaces_official_reproduction/README.md`](docs/worklines/molmospaces_official_reproduction/README.md) | Upstream MolmoSpaces entrypoints | [`results/workline_index/molmospaces_official_reproduction.md`](results/workline_index/molmospaces_official_reproduction.md) | Completed |
| MimicGen single-arm groundwork | [`docs/worklines/mimicgen_single_arm/README.md`](docs/worklines/mimicgen_single_arm/README.md) | 已由 [`src/pnp/`](src/pnp/) 继承 | [`results/workline_index/mimicgen_single_arm.md`](results/workline_index/mimicgen_single_arm.md) | Historical |

同时包含：上游 MolmoSpaces 源码快照、`results/` 下的轻量 manifest 和 summary、`media/` 下的演示媒体、公开文档和 attribution 文件。

大体积运行数据不提交到 Git：官方 shards、生成的 HDF5、rollout 目录、simulator logs、PID 文件、cache、视频、本地机器路径和内部 planning ledger。使用此类文件的 workline 会在仓库中保留 README、轻量 inventory 或再生成入口，而非提交原始产物。

## Quick Start

Clone 仓库，安装 MolmoSpaces 和依赖，放置官方 MolmoBot-Data shard，运行 smoke check。

### 1. Clone

```bash
git clone https://github.com/yanggoumao2/molmospaces-integration.git
cd molmospaces-integration
```

也可以使用 SSH：

```bash
git clone git@github.com:yanggoumao2/molmospaces-integration.git
cd molmospaces-integration
```

### 2. 创建 Python 环境

MolmoSpaces 使用 Python 3.11。

使用 conda：

```bash
conda create -n molmospaces-integration python=3.11
conda activate molmospaces-integration
python -m pip install --upgrade pip setuptools wheel
```

使用 `uv`：

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
```

### 3. 安装 MolmoSpaces

MuJoCo Pick-and-Place 集成通常使用：

```bash
pip install -e ".[mujoco]"
```

需要上游可选模块时可以安装 extras，例如：

```bash
pip install -e ".[mujoco,grasp,housegen]"
```

上游 MolmoSpaces 安装细节见 `docs/upstream_molmospaces_readme.md`。

### 4. 拉取 MimicGen 和 robomimic

集成脚本依赖 MimicGen 和 robomimic。把两者拉到 `vendor/`：

```bash
bash tools/setup_mimicgen_dependency.sh
```

如果有 package metadata，可以 editable 安装：

```bash
pip install -e vendor/robomimic
pip install -e vendor/mimicgen
```

设置依赖路径：

```bash
export MIMICGEN_ROOT=$PWD/vendor/mimicgen
export ROBOMIMIC_ROOT=$PWD/vendor/robomimic
```

如果使用已有本地 checkout，把 `MIMICGEN_ROOT` 和 `ROBOMIMIC_ROOT` 指向对应目录即可。

### 5. 设置运行路径

```bash
export MOLMOSPACES_ROOT=$PWD
export MOLMOSPACES_PYTHON=python
export MOLMOSPACES_PNP_WORKDIR=$PWD/runtime/mimicgen_pick_and_place

mkdir -p "$MOLMOSPACES_PNP_WORKDIR"/{artifacts/seeds,artifacts/mimicgen_pnp,data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards,logs}
```

可选 NLTK cache：

```bash
export MOLMOSPACES_NLTK_DATA=/path/to/nltk_data
```

### 6. 放置 MolmoBot-Data shard

下载官方 MolmoBot Pick-and-Place validation shard，放到：

```text
runtime/mimicgen_pick_and_place/data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar
```

本仓库包含轻量 manifest 和摘要，不包含官方数据 shard 或生成产物。

### 7. 运行 smoke check

把 manifest 复制到运行目录：

```bash
cp results/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json \
  "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/"
```

检查源轨迹候选，查看哪些 MolmoBot-Data 轨迹可以用作 source：

```bash
$MOLMOSPACES_PYTHON src/pnp/inspect_source_candidates.py
```

回放一条源轨迹，验证回放管线正常工作：

```bash
$MOLMOSPACES_PYTHON src/pnp/replay_source_episode.py --seed-index 0 --save-videos
```

## MolmoSpaces 工作线合集

上方的工作线表格同步维护在 [`docs/worklines/README.md`](docs/worklines/README.md) 中。每条 workline 有独立的 README 详细文档，即使其原始 HDF5/视频不在 Git 中。

MolmoAct2 官方 `sim_eval` 成功、MolmoSpaces adapter 诊断、bimanual 浏览器遥操作、custom-scene YAM baseline、iTHOR source-demo 基础设施和 Pick-and-Place MimicGen rollout 是独立的证据层级，不应混为一谈。

## Pick-and-Place 集成流程

Pick-and-Place pipeline 从 MolmoBot-Data 源轨迹生成 MimicGen rollout，流程如下：

1. 检查或准备 source candidate 元数据；
2. 回放 source trajectory 并收集 MimicGen datagen 信息；
3. 把选中 source 转成 robomimic/MimicGen source HDF5；
4. 用 MimicGen 解析 source HDF5；
5. 在 MolmoSpaces 中生成 rollout 并保存视频；
6. 可选：用 action-hash 去重收集 accepted rollouts。

### 收集 MimicGen datagen 信息

回放源轨迹，提取 MimicGen 空间变换所需的观测和动作数据：

```bash
$MOLMOSPACES_PYTHON src/pnp/collect_homogeneous_datagen_info.py \
  --seed-index 0 \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_homogeneous_foodlike_bowl_10candidate_v3"
```

### 转换为 robomimic/MimicGen source HDF5

把选中的源轨迹打包成一个 robomimic 兼容的 HDF5 文件：

```bash
$MOLMOSPACES_PYTHON src/pnp/convert_seed_set_to_robomimic.py \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5"
```

### 用 MimicGen 解析 source HDF5

将 source HDF5 加载到 MimicGen 的 dataset 格式并查看属性：

```bash
PNP_SOURCE_HDF5="$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5" \
$MOLMOSPACES_PYTHON src/pnp/parse_source_dataset.py
```

### 生成 MolmoSpaces rollout

用 MimicGen 的 object-centric 空间变换，将源轨迹迁移到新场景布局：

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

### 50-demo cross-subtask MimicGen 路线

以上流程每个 rollout 使用一条源 demo。cross-subtask 路线使用更大的 50-demo 源池，调用 MimicGen `select_src_per_subtask=True`，允许同一 rollout 的不同子任务从不同源 demo 采样：

```bash
# 从官方 MolmoBot shard 选取源池。
$MOLMOSPACES_PYTHON src/pnp/select_pnp_50_source_pool.py

# 收集候选的严格回放 + datagen_info。
bash src/pnp/run_collect_50cross_datagen_parallel.sh
# 或单个候选：
$MOLMOSPACES_PYTHON src/pnp/collect_datagen_info_50cross.py \
  --seed-index 0 \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_50demo_crossmix.json" \
  --out-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_50cross"

# 从 hard-pass 源构建 50-demo MimicGen source HDF5。
$MOLMOSPACES_PYTHON src/pnp/convert_seed_set_to_robomimic_50cross.py \
  --accepted all \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_50demo_crossmix.json" \
  --replay-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_50cross" \
  --out "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_50demo_crossmix_aligned.hdf5"

# 运行 select-src-per-subtask pilot。
bash src/pnp/run_50cross_selectsrc_pilot.sh
```

跨 house、物体和容器的随机混合会暴露 geometry、contact、IK 和子任务拼接兼容性问题。此路线目前为诊断性质，下一步是 compatibility-filtered cross-subtask 路线。

### 批量收集

非去重 baseline collector（收集所有满足成功条件的 rollout，可能包含重复）：

```bash
TARGET_SUCCESS=100 MAX_ATTEMPTS=800 bash src/pnp/collect_uniform_successes.sh
```

带 action-hash 去重的 high-yield collector（跳过动作序列与已收集轨迹相同的 rollout）：

```bash
PREVIOUS_COLLECTOR_RUN=logs/collect_uniform_successes_YYYYMMDD_HHMMSS \
TARGET_SUCCESS=100 MAX_ATTEMPTS=500 \
bash src/pnp/collect_unique_highyield_successes.sh
```

## Bimanual YAM 浏览器遥操作

bimanual YAM 场景的浏览器遥操作工具：只读浏览器查看器用于实时观测，键盘遥操作用于交互控制。

只读浏览器视频流：

```bash
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_viewer.py --host 127.0.0.1 --port 8765
```

键盘遥操作：

```bash
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_keyboard_teleop.py --host 127.0.0.1 --port 8765
```

浏览器遥操作路径为 bimanual YAM 场景提供控制和观测桥接。

## 当前结果快照

- **Heterogeneous Pick-and-Place whole-source generation**：`10/10` accepted generated rollouts，均满足 full rollout、final success、persistent success 和 30-step post-hold。见 `results/whole_source_transformfirst_summary.json`。
- **Homogeneous foodlike-to-bowl pilot**：严格自动成功 `13/100`；人工复核视觉成功 `15/100`，包含两个 one-frame trace-glitch case。
- **50-demo cross-subtask source pool**：`51` 个严格回放/datagen-info hard-pass 候选被筛选，`50` 个选入 `robomimic_pnp_50demo_crossmix_aligned.hdf5`，共 `9286` source action rows。见 `results/pnp_50cross_selected_hardpass_indices.json` 和 `results/robomimic_pnp_50demo_crossmix_aligned.summary.json`。
- **Broad random `select_src_per_subtask=True` pilot**：首批样本暴露了 geometry/contact/source-compatibility 问题。轻量日志和生成轨迹见 `results/50cross_selectsrc_pilot_20260727_182533/`，大体积视频/HDF5/数组不在 Git 中。
- **Uniform collector live snapshot**：见 `results/collector_uniform_summary_live.json`，未去重。
- **High-yield dedup collector live snapshot**：见 `results/collector_highyield_dedup_summary_live.json`，按 action hash 去重。

<details>
<summary>演示媒体</summary>

### 生成轨迹示例

[![Generated rollout examples](media/gif/heterogeneous_generated_examples.gif)](media/heterogeneous_generated_examples.mp4)

[打开 MP4](media/heterogeneous_generated_examples.mp4)

### 源轨迹候选

[![Foodlike source candidates](media/gif/foodlike_source_candidates.gif)](media/foodlike_source_candidates_grid.mp4)

[打开 MP4](media/foodlike_source_candidates_grid.mp4)

### Pilot 结果

[![Foodlike pilot outcomes](media/gif/foodlike_pilot_outcomes.gif)](media/foodlike_pilot_outcomes.mp4)

[打开 MP4](media/foodlike_pilot_outcomes.mp4)

</details>

## 仓库结构

```text
molmo_spaces/             上游 MolmoSpaces Python 包
scripts/                  上游 MolmoSpaces 脚本
configs/, examples/, docs/ 上游配置、示例和文档
src/pnp/                  Pick-and-Place MimicGen 集成脚本
src/bimanual_yam/         bimanual YAM 诊断、浏览器可视化和键盘遥操作
results/                  轻量 JSON manifest 和结果摘要
media/                    GitHub README 使用的小体积演示视频
docs/experiments.md       实验流程、结果边界和证据说明
docs/upstream_molmospaces_readme.md  上游 MolmoSpaces 原始 README
tools/setup_mimicgen_dependency.sh   拉取 MimicGen 和 robomimic 到 vendor/ 的辅助脚本
```

## 证据边界

- Source trajectories 是 MolmoBot-Data 中的 synthetic planner expert trajectories，不是 human demonstrations。
- Replay success 和 parser success 是前置条件，不等于 generated-demo success。
- Accepted generated demonstrations 需要真实 MolmoSpaces simulator rollout、`final_success=true`、post-hold stability 和保存的 artifacts。

## 作者与引用

- 上游 MolmoSpaces：Allen Institute for AI，Apache License 2.0。上游源码和 license 已保留在本仓库中。
- 上游 MimicGen：NVIDIA / NVlabs。MimicGen 论文作者为 Ajay Mandlekar、Soroush Nasiriany、Bowen Wen、Iretiayo Akinola、Yashraj Narang、Linxi Fan、Yuke Zhu 和 Dieter Fox。MimicGen 代码遵循 NVIDIA Source Code License，MimicGen 数据集遵循 CC-BY 4.0。
- MolmoSpaces Integration：Kunyu Yang，复旦大学可信具身智能研究院。
- 详细说明见 `AUTHORS.md`。

## License

上游 MolmoSpaces 代码遵循 Apache License 2.0；见 `LICENSE`。第三方依赖和数据集保留各自 license。本仓库中的集成代码按本仓库 license 作为 research code 发布，除非文件另有说明。
