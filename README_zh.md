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

**关于 `pip install -e` 的说明**：上游 MolmoSpaces `pyproject.toml` 使用的 build backend 不支持 `build_editable`（PEP 660）。如果 `pip install -e` 报错缺少 `build_editable`，包可能仍然能正常 import。如果运行脚本时出现 `ModuleNotFoundError: No module named 'molmo_spaces'`，将项目根目录加入 `PYTHONPATH`：

```bash
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

需要上游可选模块时可以安装 extras，例如：

```bash
pip install -e ".[mujoco,grasp,housegen]"
```

上游 MolmoSpaces 安装细节见 `docs/upstream_molmospaces_readme.md`。

### 3.1 安装 RB-Y1 scripted/planner datagen 所需的 CuRobo

如果你要运行官方 single-arm RB-Y1 scripted/planner data-generation config，例如 `RBY1PickDataGenConfig` 或 `RBY1PickAndPlaceDataGenConfig`，还需要安装 CuRobo extra。

这些 config 使用的是上游 package 入口：

```bash
python -m molmo_spaces.data_generation.main RBY1PickDataGenConfig
python -m molmo_spaces.data_generation.main RBY1PickAndPlaceDataGenConfig
```

它们与本仓库公开的 `src/pnp/` MimicGen 集成脚本是不同路径。不要把 `src/pnp/` 误当成官方 RB-Y1 scripted/planner datagen 入口的直接替代。

上游 `pyproject.toml` 中对应的 extra 依赖为：

```text
nvidia-curobo @ git+https://github.com/allenai/curobo.git@87e857d46fa5398f268c7f31d26566351be8671d
```

在某些机器上，`pip install -e ".[mujoco,curobo]"` 可能失败，因为 pip build isolation 会尝试额外下载较大的 `torch` wheel。这种情况下，更稳妥的方式是针对当前已激活环境安装 CuRobo，而不是让 pip 创建 isolated build env：

```bash
PIP_NO_BUILD_ISOLATION=1 pip install --no-build-isolation \
  "nvidia-curobo @ git+https://github.com/allenai/curobo.git@87e857d46fa5398f268c7f31d26566351be8671d"
```

如果网络不稳定，pip 在下载传递依赖时频繁中断，可以先安装缺失 wheel，再关闭依赖解析重试 CuRobo：

```bash
pip install embreex rtree pycollada colorlog manifold3d mapbox_earcut svg.path \
  vhacdx yourdfpy pybind11 setuptools_scm vcs-versioning warp-lang==1.11.1 \
  importlib_resources

PIP_NO_BUILD_ISOLATION=1 pip install --no-build-isolation --no-deps \
  "nvidia-curobo @ git+https://github.com/allenai/curobo.git@87e857d46fa5398f268c7f31d26566351be8671d"
```

安装后，先验证 planner 依赖，再尝试 RB-Y1 datagen：

```bash
python -c "import curobo; print(curobo.__file__)"
python - <<'PY'
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import RBY1PickAndPlaceDataGenConfig
cfg = RBY1PickAndPlaceDataGenConfig()
print(type(cfg).__name__)
print("house_inds:", cfg.task_sampler_config.house_inds)
print("samples_per_house:", cfg.task_sampler_config.samples_per_house)
print("episodes_per_batch:", cfg.task_sampler_config.episodes_per_batch)
PY
```

如果 `import curobo` 失败，RB-Y1 planner config 会在 `model_post_init()` 阶段失败，此时官方 single-arm scripted-expert pipeline 还不能在该机器上运行。

### 3.2 官方 RB-Y1 datagen 的 planner-server 说明

官方 RB-Y1 scripted/planner datagen 路径使用带 planner 的 policy config。仓库中的 tests 和 helper code 表明，这条路径既支持配置好的 planner server URL，也支持本地 planner-server workflow。

对复现者来说，真正重要的边界是：

- 先验证 `curobo` 能 import，目标 config 能成功实例化；
- 再根据你自己的环境使用合适的 planner-server 连接配置；
- 只有当 config 构造和 planner-server 连通性都在你的机器上验证通过后，才应认为 datagen 路径具备可运行前提。

由于 planner-server 的部署方式依赖具体环境，本 README 不把某个特定 host 或某种仅限本地机器的做法硬编码成唯一标准步骤。

### 4. 拉取固定版本的 MimicGen 和 robomimic

`vendor/` 不直接提交到 Git。请用脚本拉取本工作线实际使用的 upstream commit：

```bash
bash tools/setup_mimicgen_dependency.sh
```

当前 pin：

- MimicGen: `72bd767c255545f462e7ccfb2731f2e5d4c1d9bb`
- robomimic: `e10526b9a40c78b41f1e37e60041dc0ec0a5f60f`

editable 安装：

```bash
pip install -e vendor/robomimic
pip install -e vendor/mimicgen
```

设置依赖路径和 `PYTHONPATH`：

```bash
export MIMICGEN_ROOT=$PWD/vendor/mimicgen
export ROBOMIMIC_ROOT=$PWD/vendor/robomimic
export PYTHONPATH=$PWD:$MIMICGEN_ROOT:$ROBOMIMIC_ROOT:${PYTHONPATH:-}
```

如果使用已有本地 checkout，把 `MIMICGEN_ROOT` 和 `ROBOMIMIC_ROOT` 指向对应目录即可。

### 5. 设置运行路径

```bash
export MOLMOSPACES_ROOT=$PWD
export MOLMOSPACES_PYTHON=python
export MOLMOSPACES_PNP_WORKDIR=$PWD/runtime/mimicgen_pick_and_place
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export NLTK_DATA=${NLTK_DATA:-$HOME/nltk_data}
export MOLMOSPACES_NLTK_DATA=$NLTK_DATA

mkdir -p "$MOLMOSPACES_PNP_WORKDIR"/{artifacts/seeds,artifacts/mimicgen_pnp,data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards,logs}
```

`HF_HOME` 会被 robomimic 的 CLIP language embedding 工具使用。`NLTK_DATA` / `MOLMOSPACES_NLTK_DATA` 用于 MolmoSpaces 需要本地 WordNet cache 的情况。

### 6. 放置 MolmoBot-Data shard

下载官方 MolmoBot Pick-and-Place validation shard，放到：

```text
runtime/mimicgen_pick_and_place/data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar
```

本仓库包含轻量 manifest 和摘要，不包含官方数据 shard 或生成产物。

### 7. 首次运行资源 cache 说明

MolmoSpaces 首次运行时可能下载/解压 iTHOR assets。如果中途被打断，可能出现类似错误：

```text
Directory path exists on disk but is not recorded in the cache manifest
```

不要手改 manifest。把报错中提到的未注册资源目录移动到 backup，再重跑，让资源管理器重新解压并登记。例如：

```bash
mkdir -p "$HOME/.cache/molmo-spaces-resources_broken_backup"
mv "$HOME/.cache/molmo-spaces-resources/objects/thor/20251117" \
  "$HOME/.cache/molmo-spaces-resources_broken_backup/thor_20251117_$(date +%Y%m%d_%H%M%S)"
```

如果网络需要代理，请在首次下载 asset/model 前设置代理环境变量。

### 8. 运行 smoke check

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

bimanual YAM 场景的浏览器键盘遥操作工具。公开复现推荐入口是键盘遥操作 bridge，它会先执行当前 strict tabletop 初始化，再启动浏览器 UI。旧的只读 viewer 脚本不是此工作线推荐复现入口。

### 运行环境要求

- **需要 GPU 渲染**才能达到可用帧率。CPU 软件渲染（OSMesa）三相机模式下仅 ~3 FPS，键盘控制响应延迟严重。
- **NVIDIA EGL headless 渲染**是已验证的配置：Linux 主机 + NVIDIA GPU + NVIDIA 闭源驱动 + EGL vendor 文件（`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`）。遥操作工具通过 MolmoSpaces 自动使用 `MUJOCO_GL=egl`。
- **WSL2 不支持**此工作线。WSL2 使用 Mesa EGL，不暴露 `EGL_EXT_platform_device` 扩展——这是 MolmoSpaces headless GPU 渲染依赖的扩展。如果使用 WSL2，请在远程 Linux GPU 服务器上运行（见下方说明）。

### 本地运行（Linux + NVIDIA GPU）

启动遥操作 bridge：

```bash
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

终端打印本地 teleoperation URL 后打开 `http://127.0.0.1:8765`。

### 在远程 GPU 服务器上运行

如果 teleop 运行在远程 Linux GPU 服务器上，需要建立 SSH tunnel 将浏览器端口转发到本地机器：

```bash
# 在本地机器上执行：
ssh -L 8765:127.0.0.1:8765 user@your-gpu-server
```

然后在本地浏览器打开 `http://127.0.0.1:8765`。

### 浏览器键位

先点击页面；`Tab` 切换 active arm；`W/S/A/D` 在视觉平面移动；`E/Q` 上下；方向键控制 pitch/yaw；`Z/C` roll；`F` 切换当前夹爪。如果初始化失败，增大 `--initialization-max-attempts` 或查看 `runtime/bimanual_yam_initialization_report.json`。

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


## 使用新生成的 Franka PnP HDF5 作为 MimicGen 输入

原有 MolmoBot shard 输入仍是默认方式。若要把本地新生成的 Franka Pick-and-Place HDF5 接入同一 source manifest，先查看固定身份 pool，再用不同 seed 运行独立任务：

```bash
python scripts/datagen/run_pipeline.py --list_pools

export CUDA_VISIBLE_DEVICES=1

$MOLMOSPACES_PYTHON scripts/datagen/run_pipeline.py \
  --robot droid --policy planner --task_type pick_and_place \
  --pool molmodata_potato_bowl_1716 \
  --samples_per_house 10 --randomize_fixed_pickup_pose \
  --filter_for_successful_trajectories \
  --disable_action_noise --require_clean_success \
  --device gpu --num_workers 1 \
  --seed 111 --run_name_prefix potato_bowl_1716_seed111
```

每个 pool 会固定 scene dataset、split、house、pickup object 和 receptacle。不同 pool 必须使用独立输出目录和独立 source HDF5，不能混合计数。每次使用未使用过的 seed 和不同 prefix。`samples_per_house` 是请求保存的轨迹目标数，不是成功保证；每个 run 都要核验 `Success count`、身份元数据、无重试且单调的 planner phases、视频和 HDF5 内实际 trajectory 数。`--device gpu` 只将 Warp parallel IK 切到 CUDA，MuJoCo physics 仍在 CPU。`--num_workers` 控制独立 work item，不能把单 house pool 自动拆分给多个 worker。

选择一个 run 根目录，或包含多个 run 的共同父目录：

```bash
PNP_SELECT_N=50 $MOLMOSPACES_PYTHON src/pnp/select_pnp_50_source_pool.py \
  --franka-datagen-root /path/to/datagen/pick_and_place_planner_v1
```

不传该选项时，selector 仍按原逻辑读取 MolmoBot shard。Franka 模式递归读取 `house_*/trajectories_batch_*.h5`，要求 terminal/persistent success、完整 `0..9` planner phases、末帧 `task_info.success=true`，以及现有 replay/conversion 所需字段；同时按初始任务状态与动作序列的组合指纹去重。输出仍为兼容的 `pnp_seed_manifest_50demo_crossmix.json`，因此后续 datagen-info 与 conversion 命令不变。

这批数据的准确 provenance 是 **synthetic scripted-IK planner expert demos**，不是 human demonstrations，也不是 RB-Y1 CuRobo planner-server trajectories。训练前还必须检查 wrist/exocentric 视频、replay、完整 Pick → grasp/lift → transport → place/release 行为、schema/timing，并确认恰好 50 条唯一且通过验收的轨迹。
