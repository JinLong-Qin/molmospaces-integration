"""Isolated official-sim_eval MolmoAct2 YAM box bridge config.

This config keeps the older MolmoSpaces approximation route intact while loading
MolmoAct2's official flattened YAM MJCF from ``sim_eval/assets``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from molmo_spaces.configs.camera_configs import (
    CameraSystemConfig,
    RobotMountedCameraConfig,
)
from molmo_spaces.configs.policy_configs_baselines import MolmoAct2YamPolicyConfig
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, BimanualYamRobotConfig
from molmo_spaces.configs.task_configs import PackingTaskConfig
from molmo_spaces.configs.task_sampler_configs import PackingTaskSamplerConfig
from molmo_spaces.data_generation.config.molmoact2_official_yam_box_config import (
    OFFICIAL_MOLMOACT2_YAM_INSTRUCTION,
    OFFICIAL_YAM_BOX_HEIGHT,
    OFFICIAL_YAM_BOX_INNER_HALF,
    OFFICIAL_YAM_BOX_POS_XY,
    OFFICIAL_YAM_BOX_WALL,
    OFFICIAL_YAM_CAMERA_SPECS,
    OFFICIAL_YAM_OBJECT_ANCHORS_XY,
    OFFICIAL_YAM_OBJECT_NAMES,
    OFFICIAL_YAM_OPEN_BOX_NAME,
    OFFICIAL_YAM_ROBOT_WORLD_POS,
    MolmoAct2OfficialYamBoxDataGenConfig,
    MolmoAct2OfficialYamBoxTask,
    MolmoAct2OfficialYamBoxTaskSampler,
    _identity_pose_at,
    _sample_official_yam_object_pose,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.robots.robot_views.official_sim_eval_bimanual_yam_view import (
    OFFICIAL_YAM_GRIPPER_CLOSED_COMMAND,
    OFFICIAL_YAM_GRIPPER_OPEN_COMMAND,
    OfficialSimEvalBimanualYamRobotView,
)
from molmo_spaces.utils.pose import pose_mat_to_7d


def _resolve_official_molmoact2_repo_root() -> Path:
    env_root = os.environ.get("MOLMOACT2_REPO_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())

    artifact_suffix = Path(
        "artifacts/molmospaces/molmoact2_yam_integration_analysis_20260620_142443/molmoact2"
    )
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / artifact_suffix)

    candidates.append(
        Path(
            "/home/c/project/paper_reproductions/official/artifacts/molmospaces/"
            "molmoact2_yam_integration_analysis_20260620_142443/molmoact2"
        )
    )

    for candidate in candidates:
        xml_path = candidate / "sim_eval/assets/yam/yam_mujoco/bimanual_yam_linear_flattened.xml"
        if xml_path.exists():
            return candidate
    return candidates[0]


OFFICIAL_MOLMOACT2_REPO_ROOT = _resolve_official_molmoact2_repo_root()
OFFICIAL_SIM_EVAL_YAM_ASSET_DIR = OFFICIAL_MOLMOACT2_REPO_ROOT / "sim_eval/assets/yam/yam_mujoco"
OFFICIAL_SIM_EVAL_YAM_XML = OFFICIAL_SIM_EVAL_YAM_ASSET_DIR / "bimanual_yam_linear_flattened.xml"


def _hfov_degrees_to_vfov_degrees(hfov_degrees: float, resolution: tuple[int, int]) -> float:
    width, height = resolution
    focal_length = (width / 2.0) / np.tan(np.deg2rad(hfov_degrees / 2.0))
    return float(np.rad2deg(2.0 * np.arctan((height / 2.0) / focal_length)))


class MolmoAct2OfficialSimEvalYamBoxCameraSystem(CameraSystemConfig):
    """Official sim_eval YAM camera triad with MolmoSpaces vertical-FOV semantics."""

    img_resolution: tuple[int, int] = (640, 360)
    cameras: list[RobotMountedCameraConfig] = [
        RobotMountedCameraConfig(
            name="top_cam",
            reference_body_names=["robot_0/bimanual_base", "robot_0/base"],
            camera_offset=OFFICIAL_YAM_CAMERA_SPECS["top_cam"]["p"],
            camera_quaternion=OFFICIAL_YAM_CAMERA_SPECS["top_cam"]["molmospaces_q"],
            fov=_hfov_degrees_to_vfov_degrees(
                OFFICIAL_YAM_CAMERA_SPECS["top_cam"]["hfov_deg"],
                OFFICIAL_YAM_CAMERA_SPECS["top_cam"]["resolution"],
            ),
            visibility_constraints={"__official_yam_task_objects__": 0.0001},
        ),
        RobotMountedCameraConfig(
            name="left_cam",
            reference_body_names=["robot_0/left_link_6"],
            camera_offset=OFFICIAL_YAM_CAMERA_SPECS["left_cam"]["p"],
            camera_quaternion=OFFICIAL_YAM_CAMERA_SPECS["left_cam"]["molmospaces_q"],
            fov=_hfov_degrees_to_vfov_degrees(
                OFFICIAL_YAM_CAMERA_SPECS["left_cam"]["hfov_deg"],
                OFFICIAL_YAM_CAMERA_SPECS["left_cam"]["resolution"],
            ),
        ),
        RobotMountedCameraConfig(
            name="right_cam",
            reference_body_names=["robot_0/right_link_6"],
            camera_offset=OFFICIAL_YAM_CAMERA_SPECS["right_cam"]["p"],
            camera_quaternion=OFFICIAL_YAM_CAMERA_SPECS["right_cam"]["molmospaces_q"],
            fov=_hfov_degrees_to_vfov_degrees(
                OFFICIAL_YAM_CAMERA_SPECS["right_cam"]["hfov_deg"],
                OFFICIAL_YAM_CAMERA_SPECS["right_cam"]["resolution"],
            ),
        ),
    ]


OFFICIAL_SIM_EVAL_SCENE_Z_OFFSET = 0.0
OFFICIAL_SIM_EVAL_BOX_WORLD_Z = 0.0
OFFICIAL_SIM_EVAL_BOX_FLOOR_TOP_Z = OFFICIAL_SIM_EVAL_BOX_WORLD_Z + OFFICIAL_YAM_BOX_WALL
OFFICIAL_SIM_EVAL_BOX_RIM_Z = OFFICIAL_SIM_EVAL_BOX_FLOOR_TOP_Z + OFFICIAL_YAM_BOX_HEIGHT


def official_sim_eval_yam_box_success_from_positions(
    object_positions: dict[str, np.ndarray],
) -> dict[str, Any]:
    bx, by = OFFICIAL_YAM_BOX_POS_XY
    per_object: dict[str, dict[str, Any]] = {}
    n_in_box = 0
    for name in OFFICIAL_YAM_OBJECT_NAMES:
        pos = np.asarray(object_positions[name], dtype=float).reshape(3)
        dx = abs(float(pos[0]) - bx)
        dy = abs(float(pos[1]) - by)
        inside_xy = dx < OFFICIAL_YAM_BOX_INNER_HALF and dy < OFFICIAL_YAM_BOX_INNER_HALF
        z_ok = (
            pos[2] > OFFICIAL_SIM_EVAL_BOX_FLOOR_TOP_Z - 0.01
            and pos[2] < OFFICIAL_SIM_EVAL_BOX_RIM_Z + 0.05
        )
        in_box = bool(inside_xy and z_ok)
        n_in_box += int(in_box)
        per_object[name] = {
            "position": pos.tolist(),
            "dx_from_box_center": dx,
            "dy_from_box_center": dy,
            "inside_xy": bool(inside_xy),
            "z_ok": bool(z_ok),
            "in_box": in_box,
        }
    return {
        "success": n_in_box == len(OFFICIAL_YAM_OBJECT_NAMES),
        "n_in_box": n_in_box,
        "n_total": len(OFFICIAL_YAM_OBJECT_NAMES),
        "per_object": per_object,
    }


class MolmoAct2OfficialSimEvalYamBoxTask(MolmoAct2OfficialYamBoxTask):
    """Official low-scene Z metric for the sim_eval MJCF bridge."""

    def evaluate_official_yam_success(self, batch_index: int = 0) -> dict[str, Any]:
        return official_sim_eval_yam_box_success_from_positions(self._object_positions(batch_index))


class MolmoAct2OfficialSimEvalYamBoxTaskSampler(MolmoAct2OfficialYamBoxTaskSampler):
    """Sampler variant that keeps the old route intact but uses official low-scene Z."""

    def randomize_scene(self, env, robot_view) -> None:
        super().randomize_scene(env, robot_view)
        model, data = env.current_model, env.current_data
        robot_view.base.pose = _identity_pose_at(OFFICIAL_YAM_ROBOT_WORLD_POS)
        for name, anchor_xy in OFFICIAL_YAM_OBJECT_ANCHORS_XY.items():
            obj = create_mlspaces_body(data, name)
            object_z = float(obj.position[2]) + OFFICIAL_SIM_EVAL_SCENE_Z_OFFSET
            obj.pose = _sample_official_yam_object_pose(anchor_xy, object_z)
        mujoco.mj_forward(model, data)

    def get_workspace_center(self, env) -> np.ndarray:
        points = [
            np.array([*xy, OFFICIAL_SIM_EVAL_BOX_FLOOR_TOP_Z], dtype=float)
            for xy in OFFICIAL_YAM_OBJECT_ANCHORS_XY.values()
        ]
        points.append(
            np.array([*OFFICIAL_YAM_BOX_POS_XY, OFFICIAL_SIM_EVAL_BOX_RIM_Z], dtype=float)
        )
        return np.mean(points, axis=0)

    def _sample_task(self, env) -> MolmoAct2OfficialSimEvalYamBoxTask:
        task_cfg = self.config.task_config
        task_cfg.pickup_obj_name = OFFICIAL_YAM_OBJECT_NAMES[0]
        task_cfg.place_receptacle_name = OFFICIAL_YAM_OPEN_BOX_NAME
        task_cfg.referral_expressions = {
            "pickup_name": "everything",
            "place_name": "box",
            "instruction": OFFICIAL_MOLMOACT2_YAM_INSTRUCTION,
        }
        task_cfg.added_objects = {
            name: Path("examples/molmoact2_official_sim_eval_yam_box/scene.xml")
            for name in (*OFFICIAL_YAM_OBJECT_NAMES, OFFICIAL_YAM_OPEN_BOX_NAME)
        }
        data = env.current_data
        task_cfg.object_poses = {
            name: pose_mat_to_7d(create_mlspaces_body(data, name).pose).tolist()
            for name in (*OFFICIAL_YAM_OBJECT_NAMES, OFFICIAL_YAM_OPEN_BOX_NAME)
        }
        task_cfg.pickup_obj_start_pose = task_cfg.object_poses[OFFICIAL_YAM_OBJECT_NAMES[0]]
        task_cfg.place_receptacle_start_pose = task_cfg.object_poses[OFFICIAL_YAM_OPEN_BOX_NAME]
        task_cfg.robot_base_pose = pose_mat_to_7d(env.current_robot.robot_view.base.pose).tolist()
        self.setup_cameras(env, deterministic_only=False)
        return MolmoAct2OfficialSimEvalYamBoxTask(env, self.config)


@register_config("MolmoAct2OfficialSimEvalYamBoxDataGenConfig")
class MolmoAct2OfficialSimEvalYamBoxDataGenConfig(MolmoAct2OfficialYamBoxDataGenConfig):
    """Official-MJCF bridge for ``BimanualYAMPutEverythingInBox-v1`` diagnostics."""

    seed: int | None = 42
    end_on_success: bool = True
    environment_light_intensity: float = 6000.0
    robot_config: BimanualYamRobotConfig = BimanualYamRobotConfig(
        robot_view_factory=OfficialSimEvalBimanualYamRobotView,
        robot_dir=OFFICIAL_SIM_EVAL_YAM_ASSET_DIR,
        robot_xml_path=Path("bimanual_yam_linear_flattened.xml"),
        base_size=None,
        name="official_sim_eval_bimanual_yam",
        init_qpos={
            # Matches official BimanualYAM home keyframe used by
            # sim_eval/tasks/yam_tasks/bimanual_put_everything_in_box.py
            # (set_qpos(agent.keyframes["home"].qpos) where home is
            # zeros(16)). The rest keyframe is NOT used at reset.
            "left_arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "right_arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "left_gripper": [0.0, 0.0],
            "right_gripper": [0.0, 0.0],
        },
        init_qpos_noise_range=None,
        # ManiSkill rebuilds the MJCF actuators with these PD gains and force
        # limits in BimanualYAM._controller_configs. Keep MuJoCo equivalent.
        K_stiffness=[
            40.0,
            40.0,
            40.0,
            20.0,
            10.0,
            10.0,
            2000.0,
            2000.0,
            40.0,
            40.0,
            40.0,
            20.0,
            10.0,
            10.0,
            2000.0,
            2000.0,
        ],
        K_damping=[
            2.5,
            2.5,
            2.5,
            0.5,
            1.0,
            1.0,
            40.0,
            40.0,
            2.5,
            2.5,
            2.5,
            0.5,
            1.0,
            1.0,
            40.0,
            40.0,
        ],
        force_limit=[
            28.0,
            28.0,
            28.0,
            10.0,
            10.0,
            10.0,
            40.0,
            40.0,
            28.0,
            28.0,
            28.0,
            10.0,
            10.0,
            10.0,
            40.0,
            40.0,
        ],
        action_noise_config=ActionNoiseConfig(enabled=False),
    )
    task_sampler_config: PackingTaskSamplerConfig = PackingTaskSamplerConfig(
        task_sampler_class=MolmoAct2OfficialSimEvalYamBoxTaskSampler,
        dataset_name="user",
        scene_xml_paths=["examples/molmoact2_official_sim_eval_yam_box/scene.xml"],
        house_variant="base",
        house_inds=[0],
        samples_per_house=1,
        max_tasks=1,
        episodes_per_batch=1,
        task_batch_size=1,
        sim_settle_timesteps=0,
        check_robot_placement_visibility=False,
        randomize_lighting=False,
        randomize_textures=False,
        randomize_dynamics=False,
    )
    task_config: PackingTaskConfig = PackingTaskConfig(task_cls=MolmoAct2OfficialSimEvalYamBoxTask)
    camera_config: MolmoAct2OfficialSimEvalYamBoxCameraSystem = (
        MolmoAct2OfficialSimEvalYamBoxCameraSystem()
    )
    policy_config: MolmoAct2YamPolicyConfig = MolmoAct2YamPolicyConfig(
        remote_config={
            "host": "127.0.0.1",
            "port": 8203,
            "path": "/act",
            "timeout": 60.0,
        },
        endpoint_url="http://127.0.0.1:8203/act",
        instruction_override=OFFICIAL_MOLMOACT2_YAM_INSTRUCTION,
        num_steps=30,
        n_action_steps=None,
        gripper_max=abs(OFFICIAL_YAM_GRIPPER_OPEN_COMMAND),
        gripper_open_command=OFFICIAL_YAM_GRIPPER_OPEN_COMMAND,
        gripper_closed_command=OFFICIAL_YAM_GRIPPER_CLOSED_COMMAND,
        gripper_scale_source="official_sim_eval_bimanual_yam_linear_flattened_xml_ctrlrange",
        debug_dump_dir="artifacts/molmospaces/molmoact2_official_sim_eval_yam_box_debug",
        debug_dump_max_calls=2,
        raw_action_log_path=(
            "artifacts/molmospaces/molmoact2_official_sim_eval_yam_box_debug/raw_actions.jsonl"
        ),
        execution_mode="sim_eval_step",
        execution_command_hz=30.0,
        camera_mapping={
            "top_cam": "top_cam",
            "left_cam": "left_cam",
            "right_cam": "right_cam",
        },
    )

    @property
    def tag(self) -> str:
        return "molmoact2_official_sim_eval_yam_box"
