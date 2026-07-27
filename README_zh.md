<div align="center">
  <img src="media/banner.svg" alt="MolmoSpaces x MimicGen" width="980" />
</div>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <strong>MolmoSpaces x MimicGen</strong> 把上游 MolmoSpaces 代码库和一个 MimicGen 集成层放在同一个仓库中，用于 Pick-and-Place 轨迹生成，以及 bimanual YAM 的浏览器可视化和键盘遥操作。
</p>

<p align="center">
  本仓库目标是可复现并持续维护。新用户可以 clone 仓库，安装 MolmoSpaces，拉取 MimicGen 和 robomimic，把官方 MolmoBot-Data shard 放到 <code>runtime/</code>，然后运行 source replay、datagen-info、source-HDF5 转换、rollout generation 和 collection 脚本。
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

<p align="center"><em>顶部预览保持紧凑；点击 MP4 链接可查看完整尺寸演示。</em></p>

## 包含内容

这个仓库是 **MolmoSpaces integration workline portfolio**，不是只有 Pick-and-Place 的单线 release。首页直接列出全部公开 workline，避免读者误以为只有少数几个方向。

| Workline | Canonical README | 代码入口 | 证据 / inventory | 状态 |
|---|---|---|---|---|
| MimicGen Pick-and-Place | [`docs/worklines/mimicgen_pick_and_place/README.md`](docs/worklines/mimicgen_pick_and_place/README.md) | [`src/pnp/`](src/pnp/) | [`results/workline_index/mimicgen_pick_and_place.md`](results/workline_index/mimicgen_pick_and_place.md) | Active / primary |
| 50-demo MimicGen cross-subtask route | [`docs/worklines/mimicgen_50cross/README.md`](docs/worklines/mimicgen_50cross/README.md) | [`src/pnp/*50cross*`](src/pnp/) | [`results/50cross_selectsrc_pilot_20260727_182533/`](results/50cross_selectsrc_pilot_20260727_182533/) | Diagnostic |
| Bimanual YAM browser teleoperation | [`docs/worklines/bimanual_yam_browser_teleop/README.md`](docs/worklines/bimanual_yam_browser_teleop/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | Infrastructure |
| iTHOR bimanual YAM | [`docs/worklines/ithor_bimanual_yam/README.md`](docs/worklines/ithor_bimanual_yam/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | In progress |
| Completed custom-scene bimanual YAM baseline | [`docs/worklines/bimanual_yam_source_baseline/README.md`](docs/worklines/bimanual_yam_source_baseline/README.md) | Inventory / regeneration entrypoints | [`results/workline_index/bimanual_yam_source_baseline.md`](results/workline_index/bimanual_yam_source_baseline.md) | Completed evidence package |
| MolmoAct2 → MolmoSpaces legacy | [`docs/worklines/molmoact2_legacy/README.md`](docs/worklines/molmoact2_legacy/README.md) | [`src/molmoact2_legacy/`](src/molmoact2_legacy/) | [`results/workline_index/molmoact2_integration_legacy.md`](results/workline_index/molmoact2_integration_legacy.md) | Legacy / ended |
| Bounded official MolmoSpaces reproduction | [`docs/worklines/molmospaces_official_reproduction/README.md`](docs/worklines/molmospaces_official_reproduction/README.md) | Upstream MolmoSpaces entrypoints | [`results/workline_index/molmospaces_official_reproduction.md`](results/workline_index/molmospaces_official_reproduction.md) | Completed bounded reproduction |
| MimicGen single-arm groundwork | [`docs/worklines/mimicgen_single_arm/README.md`](docs/worklines/mimicgen_single_arm/README.md) | 已由 [`src/pnp/`](src/pnp/) 继承 | [`results/workline_index/mimicgen_single_arm.md`](results/workline_index/mimicgen_single_arm.md) | Historical groundwork |

同时包含：上游 MolmoSpaces 源码快照、`results/` 下的轻量 manifest/summary、`media/` 下的小体积 README 演示媒体、公开文档和 attribution 文件。

仓库刻意不提交大体积运行数据：官方 shards、生成的 HDF5、rollout 目录、simulator logs、PID 文件、cache、视频、本地机器路径和内部 planning ledger。每条 workline 至少保留 README、轻量 inventory 或再生成入口，而不是只给少数方向单独文档。

## Quick Start

### 1. Clone

```bash
git clone https://github.com/yanggoumao2/molmospaces-mimicgen.git
cd molmospaces-mimicgen
```

也可以使用 SSH：

```bash
git clone git@github.com:yanggoumao2/molmospaces-mimicgen.git
cd molmospaces-mimicgen
```

### 2. 创建 Python 环境

MolmoSpaces 使用 Python 3.11。

使用 conda：

```bash
conda create -n molmospaces-mimicgen python=3.11
conda activate molmospaces-mimicgen
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

集成脚本使用 MimicGen 和 robomimic。把两者拉到 `vendor/`：

```bash
bash tools/setup_mimicgen_dependency.sh
```

如果依赖仓库包含 package metadata，可以 editable 安装：

```bash
pip install -e vendor/robomimic
pip install -e vendor/mimicgen
```

设置依赖路径：

```bash
export MIMICGEN_ROOT=$PWD/vendor/mimicgen
export ROBOMIMIC_ROOT=$PWD/vendor/robomimic
```

如果使用已有本地 checkout，可以把 `MIMICGEN_ROOT` 和 `ROBOMIMIC_ROOT` 指向对应目录。

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

下载官方 MolmoBot Pick-and-Place validation shard，并放到：

```text
runtime/mimicgen_pick_and_place/data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar
```

本仓库包含轻量 manifest 和摘要，但不包含官方数据 shard 或生成产物。

### 7. 运行 smoke check

把 manifest 复制到运行目录：

```bash
cp results/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json \
  "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/"
```

检查源轨迹候选：

```bash
$MOLMOSPACES_PYTHON src/pnp/inspect_source_candidates.py
```

回放一条源轨迹：

```bash
$MOLMOSPACES_PYTHON src/pnp/replay_source_episode.py --seed-index 0 --save-videos
```

成功回放只证明 source trajectory replay 通过，还不等于已经生成 MimicGen rollout。

## Pick-and-Place 集成流程

Pick-and-Place pipeline 被整理成可复现的顺序流程：

1. 检查或准备 source candidate 元数据；
2. 回放 source trajectory 并收集 MimicGen datagen 信息；
3. 把选中 source 转成 robomimic/MimicGen source HDF5；
4. 用 MimicGen 解析 source HDF5；
5. 在 MolmoSpaces 中生成 rollout 并保存视频；
6. 可选：用 action-hash 去重收集 accepted rollouts。

收集 MimicGen datagen 信息：

```bash
$MOLMOSPACES_PYTHON src/pnp/collect_homogeneous_datagen_info.py \
  --seed-index 0 \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out-root "$MOLMOSPACES_PNP_WORKDIR/artifacts/replay_pnp_exact_homogeneous_foodlike_bowl_10candidate_v3"
```

把选中的源轨迹转换成 robomimic/MimicGen source HDF5：

```bash
$MOLMOSPACES_PYTHON src/pnp/convert_seed_set_to_robomimic.py \
  --manifest "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/pnp_seed_manifest_homogeneous_foodlike_bowl_10candidate_v3.json" \
  --out "$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5"
```

用 MimicGen 解析 source HDF5：

```bash
PNP_SOURCE_HDF5="$MOLMOSPACES_PNP_WORKDIR/artifacts/seeds/robomimic_pnp_foodlike_bowl_10demo_aligned.hdf5" \
$MOLMOSPACES_PYTHON src/pnp/parse_source_dataset.py
```

生成一条 rollout：

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

非去重 baseline collector：

```bash
TARGET_SUCCESS=100 MAX_ATTEMPTS=800 bash src/pnp/collect_uniform_successes.sh
```

带 action-hash 去重的 high-yield collector：

```bash
PREVIOUS_COLLECTOR_RUN=logs/collect_uniform_successes_YYYYMMDD_HHMMSS \
TARGET_SUCCESS=100 MAX_ATTEMPTS=500 \
bash src/pnp/collect_unique_highyield_successes.sh
```

## Bimanual YAM 浏览器遥操作

只读浏览器视频流：

```bash
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_viewer.py --host 127.0.0.1 --port 8765
```

键盘遥操作：

```bash
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_keyboard_teleop.py --host 127.0.0.1 --port 8765
```

浏览器遥操作路径证明的是控制和观测桥接，不等于完成任务 demonstration。

## 当前结果快照

- heterogeneous Pick-and-Place whole-source generation：`10/10` accepted generated rollouts，满足 full rollout、final success、persistent success 和 30-step post-hold。见 `results/whole_source_transformfirst_summary.json`。
- homogeneous foodlike-to-bowl pilot：严格自动成功 `13/100`；人工复核视觉成功 `15/100`，其中包含两个 one-frame trace-glitch case。
- uniform collector live snapshot：见 `results/collector_uniform_summary_live.json`，未去重。
- high-yield deduplicated collector live snapshot：见 `results/collector_highyield_dedup_summary_live.json`，按 action hash 去重。
- 仓库包含 source-pool、source replay、datagen-info、source-HDF5 转换、rollout generation 和 collection 的脚本入口；用户放置官方 shard 后可以按流程本地复现。

这些 snapshot 是进展证据。仓库会随着更多 source set、collection 脚本和 cleaned dataset 的整理继续维护。

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

- Source trajectories 是 MolmoBot-Data 里的 synthetic planner expert trajectories，不是 human demonstrations。
- Replay success 和 parser success 是前置条件，不等于 generated-demo success。
- Accepted generated demonstrations 需要真实 MolmoSpaces simulator rollout、`final_success=true`、post-hold stability 和保存的 artifacts。
- 大规模生成产物故意不放进 Git。

## 作者与引用

- 上游 MolmoSpaces：Allen Institute for AI，Apache License 2.0。上游源码和 license 已保留在本仓库中。
- 上游 MimicGen：NVIDIA / NVlabs。MimicGen 论文作者为 Ajay Mandlekar、Soroush Nasiriany、Bowen Wen、Iretiayo Akinola、Yashraj Narang、Linxi Fan、Yuke Zhu 和 Dieter Fox。MimicGen 代码遵循 NVIDIA Source Code License，MimicGen 数据集遵循 CC-BY 4.0。
- MolmoSpaces x MimicGen 集成、实验流程整理和发布组织：Kunyu Yang，复旦大学可信具身智能研究院。
- 详细说明见 `AUTHORS.md`。

## License

上游 MolmoSpaces 代码遵循 Apache License 2.0；见 `LICENSE`。第三方依赖和数据集保留各自 license。本仓库中的 MimicGen 集成代码按本仓库 license 作为 research code 发布，除非文件另有说明。
