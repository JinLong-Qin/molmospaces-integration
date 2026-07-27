"""Fixed MolmoAct2-YAM-like tabletop packing preset for MolmoSpaces.

This config intentionally avoids random iTHOR rooms. It keeps MolmoSpaces as the
simulation/evaluation pipeline while moving the scene, robot start state, and
camera triad closer to the visible MolmoAct2 YAM sim_eval/examples config:
top/left/right RGB views, a fixed tabletop object/receptacle setup, and the
YAM joint/state convention used by the MolmoAct2 /act server.
"""

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.base_packing_configs import PackingDataGenConfig
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import CameraSystemConfig, RobotMountedCameraConfig
from molmo_spaces.configs.policy_configs_baselines import MolmoAct2YamPolicyConfig
from molmo_spaces.configs.robot_configs import BimanualYamRobotConfig
from molmo_spaces.configs.task_sampler_configs import (
    PackingTaskSamplerConfig,
    PickTaskSamplerConfig,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.molmo_spaces_constants import (
    register_user_asset_library,
    register_user_grasp_library,
)
from molmo_spaces.tasks.packing_task_sampler import PackingTaskSampler
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.constants.object_constants import PICK_AND_PLACE_OBJECTS
from molmo_spaces.utils.pose import pose_mat_to_7d

_PRESET_DIR = Path(__file__).resolve().parents[3] / "examples" / "molmoact2_yam_tabletop"
register_user_asset_library("molmoact2_yam_tabletop", _PRESET_DIR / "asset_library")
register_user_grasp_library(
    "molmoact2_yam_tabletop", _PRESET_DIR / "asset_library", "molmoact2_yam_tabletop"
)


class MolmoAct2YamOfficialLikeTabletopCameraSystem(CameraSystemConfig):
    """Official-like MolmoAct2 sim_eval YAM camera triad for tabletop rollouts.

    Source: local MolmoAct2 sim_eval `robots/bimanual_yam.py` uses 640x360 RGB,
    a top camera mounted on `bimanual_base` at p=[0.15, 0, 0.8] with hfov=69.4,
    and left/right wrist cameras mounted on `link_6` at p=[0, 0.09, 0.06]
    with hfov=87. MolmoSpaces currently consumes this scalar as its camera FOV.
    """

    img_resolution: tuple[int, int] = (640, 360)
    cameras: list[RobotMountedCameraConfig] = [
        RobotMountedCameraConfig(
            name="top_cam",
            reference_body_names=["robot_0/bimanual_base", "robot_0/base"],
            camera_offset=[0.15, 0.0, 0.8],
            camera_quaternion=[0.7660444431189782, 0.0, 0.6427876096865391, 0.0],
            fov=69.4,
            visibility_constraints={"__task_objects__": 0.001},
        ),
        RobotMountedCameraConfig(
            name="left_cam",
            reference_body_names=["robot_0/left_link_6"],
            camera_offset=[0.0, 0.09, 0.06],
            camera_quaternion=[
                0.612372429196013,
                -0.35355339154618404,
                -0.3535533966987049,
                -0.612372438120441,
            ],
            fov=87.0,
        ),
        RobotMountedCameraConfig(
            name="right_cam",
            reference_body_names=["robot_0/right_link_6"],
            camera_offset=[0.0, 0.09, 0.06],
            camera_quaternion=[
                0.612372429196013,
                -0.35355339154618404,
                -0.3535533966987049,
                -0.612372438120441,
            ],
            fov=87.0,
        ),
    ]


class MolmoAct2YamVisibleTabletopCameraSystem(CameraSystemConfig):
    """Visibility-first tabletop control cameras for MolmoSpaces debugging.

    This is not an official MolmoAct2 camera distribution. It is a diagnostic
    camera triad mounted on the robot base so both the pickup object and the
    box remain visible before running longer behavior rollouts.
    """

    img_resolution: tuple[int, int] = (640, 360)
    cameras: list[RobotMountedCameraConfig] = [
        RobotMountedCameraConfig(
            name="top_cam",
            reference_body_names=["robot_0/base", "robot_0/bimanual_base"],
            camera_offset=[0.08, 0.0, 1.78],
            lookat_offset=[0.43, 0.0, 0.79],
            up_axis="z",
            fov=120.0,
            visibility_constraints={"__task_objects__": 0.001},
        ),
        RobotMountedCameraConfig(
            name="left_cam",
            reference_body_names=["robot_0/base", "robot_0/bimanual_base"],
            camera_offset=[0.06, 0.48, 1.18],
            lookat_offset=[0.44, -0.02, 0.84],
            up_axis="z",
            fov=105.0,
            visibility_constraints={"__task_objects__": 0.001},
        ),
        RobotMountedCameraConfig(
            name="right_cam",
            reference_body_names=["robot_0/base", "robot_0/bimanual_base"],
            camera_offset=[0.18, 0.42, 1.42],
            lookat_offset=[0.42, -0.02, 0.80],
            up_axis="z",
            fov=115.0,
            visibility_constraints={"__task_objects__": 0.001},
        ),
    ]


MolmoAct2YamTabletopCameraSystem = MolmoAct2YamOfficialLikeTabletopCameraSystem


class MolmoAct2YamTabletopPackingTaskSampler(PackingTaskSampler):
    """Deterministic tabletop sampler for a narrow MolmoAct2-YAM smoke rollout."""

    pickup_pose = np.array([0.43, 0.09, 0.790, 1.0, 0.0, 0.0, 0.0], dtype=float)
    box_pose = np.array([0.46, -0.14, 0.90, 0.70710678, 0.70710678, 0.0, 0.0], dtype=float)

    def _prepare_place_target(
        self,
        env: CPUMujocoEnv,
        place_target_name: str,
        pickup_obj_name: str,
        pickup_obj_pos: np.ndarray,
        supporting_geom_id: int,
    ) -> bool:
        del pickup_obj_pos, supporting_geom_id
        self.place_receptacle_name = place_target_name
        receptacle_obj = create_mlspaces_body(env.current_data, place_target_name)
        receptacle_obj.pose = _pose7d_to_mat(self.box_pose)
        mujoco.mj_forward(env.current_model, env.current_data)
        return True

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        task_cfg = self.config.task_config
        pickup_obj = create_mlspaces_body(env.current_data, task_cfg.pickup_obj_name)
        pickup_obj.pose = _pose7d_to_mat(self.pickup_pose)

        robot_view = env.current_robot.robot_view
        robot_view.base.pose = np.eye(4)

        mujoco.mj_forward(env.current_model, env.current_data)

        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()
        pickup_goal = pose_mat_to_7d(pickup_obj.pose)
        pickup_goal[2] += 0.08
        task_cfg.pickup_obj_goal_pose = pickup_goal.tolist()


class MolmoAct2YamTabletopPickTaskSampler(PickTaskSampler):
    """Deterministic YAM-like sampler for the official default `pick up the object` task.

    Evidence basis: MolmoAct2 `examples/yam/configs/yam_left.yaml` sets
    `storage.language_instruction: pick up the object`. This sampler avoids the
    unverified box-packing objective and exposes a single tabletop cube.
    """

    pickup_pose = np.array([0.43, 0.02, 0.790, 1.0, 0.0, 0.0, 0.0], dtype=float)

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        task_cfg = self.config.task_config
        pickup_obj = create_mlspaces_body(env.current_data, task_cfg.pickup_obj_name)
        pickup_obj.pose = _pose7d_to_mat(self.pickup_pose)

        robot_view = env.current_robot.robot_view
        robot_view.base.pose = np.eye(4)

        mujoco.mj_forward(env.current_model, env.current_data)

        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()
        pickup_goal = pose_mat_to_7d(pickup_obj.pose)
        pickup_goal[2] += 0.05
        task_cfg.pickup_obj_goal_pose = pickup_goal.tolist()


class MolmoAct2YamTabletopNearPathPickTaskSampler(MolmoAct2YamTabletopPickTaskSampler):
    """Near-path non-overlap pick sampler for MolmoAct2-YAM calibration.

    The cube starts on the table and away from the gripper. This is not a
    pre-grasp/overlap probe; the saved H5 must still show reach/contact/lift for
    a real success claim.
    """

    pickup_pose = np.array([0.25, -0.18, 0.790, 1.0, 0.0, 0.0, 0.0], dtype=float)


@register_config("MolmoAct2YamTabletopPickDataGenConfig")
class MolmoAct2YamTabletopPickDataGenConfig(PickBaseConfig):
    """MolmoAct2-BimanualYAM official-default pick-up task in MolmoSpaces.

    This is a distribution-alignment smoke: it mirrors the public YAM eval
    instruction (`pick up the object`) and the left/front/right camera triad,
    but the physical scene remains a MolmoSpaces approximation.
    """

    scene_dataset: str = "user"
    data_split: str = "train"
    seed: int = 22
    num_workers: int = 1
    task_horizon: int = 60
    policy_dt_ms: int = 40
    filter_for_successful_trajectories: bool = False
    output_dir: Path = Path("experiment_output")
    robot_config: BimanualYamRobotConfig = BimanualYamRobotConfig(init_qpos_noise_range=None)
    camera_config: MolmoAct2YamOfficialLikeTabletopCameraSystem = (
        MolmoAct2YamOfficialLikeTabletopCameraSystem()
    )
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=MolmoAct2YamTabletopPickTaskSampler,
        dataset_name="user",
        scene_xml_paths=["examples/molmoact2_yam_tabletop/scene.xml"],
        house_variant="base",
        house_inds=None,
        samples_per_house=1,
        episodes_per_batch=1,
        task_batch_size=1,
        pickup_types=["cube", "block"],
        filter_for_grasps=False,
        check_robot_placement_visibility=False,
        sim_settle_timesteps=20,
    )
    policy_config: MolmoAct2YamPolicyConfig = MolmoAct2YamPolicyConfig(
        instruction_override="Put everything into the box.",
        num_steps=30,
        n_action_steps=None,
        debug_dump_dir="artifacts/molmospaces/molmoact2_yam_tabletop_pick_debug",
        debug_dump_max_calls=2,
        raw_action_log_path="artifacts/molmospaces/molmoact2_yam_tabletop_pick_debug/raw_actions.jsonl",
        camera_mapping={
            "top_cam": "top_cam",
            "left_cam": "left_cam",
            "right_cam": "right_cam",
        },
    )

    @property
    def tag(self) -> str:
        return "molmoact2_yam_tabletop_pick"


@register_config("MolmoAct2YamTabletopNearPathPickDataGenConfig")
class MolmoAct2YamTabletopNearPathPickDataGenConfig(MolmoAct2YamTabletopPickDataGenConfig):
    """Near-path non-overlap variant used for a bounded fusion calibration attempt."""

    seed: int = 23
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=MolmoAct2YamTabletopNearPathPickTaskSampler,
        dataset_name="user",
        scene_xml_paths=["examples/molmoact2_yam_tabletop/scene.xml"],
        house_variant="base",
        house_inds=None,
        samples_per_house=1,
        episodes_per_batch=1,
        task_batch_size=1,
        pickup_types=["cube", "block"],
        filter_for_grasps=False,
        check_robot_placement_visibility=False,
        sim_settle_timesteps=20,
    )
    policy_config: MolmoAct2YamPolicyConfig = MolmoAct2YamPolicyConfig(
        instruction_override="pick up the object",
        num_steps=30,
        n_action_steps=None,
        debug_dump_dir="artifacts/molmospaces/molmoact2_yam_tabletop_nearpath_pick_debug",
        debug_dump_max_calls=2,
        raw_action_log_path=(
            "artifacts/molmospaces/molmoact2_yam_tabletop_nearpath_pick_debug/raw_actions.jsonl"
        ),
        camera_mapping={
            "top_cam": "top_cam",
            "left_cam": "left_cam",
            "right_cam": "right_cam",
        },
    )

    @property
    def tag(self) -> str:
        return "molmoact2_yam_tabletop_nearpath_pick"


@register_config("MolmoAct2YamTabletopPackingDataGenConfig")
class MolmoAct2YamTabletopPackingDataGenConfig(PackingDataGenConfig):
    scene_dataset: str = "user"
    data_split: str = "train"
    seed: int = 21
    num_workers: int = 1
    task_horizon: int = 60
    policy_dt_ms: int = 40
    filter_for_successful_trajectories: bool = False
    output_dir: Path = Path("experiment_output")
    robot_config: BimanualYamRobotConfig = BimanualYamRobotConfig(init_qpos_noise_range=None)
    camera_config: MolmoAct2YamVisibleTabletopCameraSystem = (
        MolmoAct2YamVisibleTabletopCameraSystem()
    )
    task_sampler_config: PackingTaskSamplerConfig = PackingTaskSamplerConfig(
        task_sampler_class=MolmoAct2YamTabletopPackingTaskSampler,
        dataset_name="user",
        scene_xml_paths=["examples/molmoact2_yam_tabletop/scene.xml"],
        house_variant="base",
        house_inds=None,
        samples_per_house=1,
        episodes_per_batch=1,
        task_batch_size=1,
        pickup_types=list(PICK_AND_PLACE_OBJECTS) + ["block", "cube"],
        filter_for_grasps=False,
        check_robot_placement_visibility=False,
        sim_settle_timesteps=20,
        box_uids=["Box_1"],
        num_place_receptacles=1,
        episodes_per_receptacle=0,
        min_object_to_receptacle_dist=0.10,
        max_object_to_receptacle_dist=0.25,
    )
    policy_config: MolmoAct2YamPolicyConfig = MolmoAct2YamPolicyConfig(
        instruction_override="Put everything into the box.",
        num_steps=30,
        n_action_steps=None,
        debug_dump_dir="artifacts/molmospaces/molmoact2_yam_tabletop_debug",
        debug_dump_max_calls=2,
        raw_action_log_path="artifacts/molmospaces/molmoact2_yam_tabletop_debug/raw_actions.jsonl",
        grasping_type="binary",
        grasping_threshold=0.5,
        camera_mapping={
            "top_cam": "top_cam",
            "left_cam": "left_cam",
            "right_cam": "right_cam",
        },
    )

    @property
    def tag(self) -> str:
        return "molmoact2_yam_tabletop_packing"


def _pose7d_to_mat(pose7d: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = pose7d[:3]
    pose[:3, :3] = R.from_quat(pose7d[3:], scalar_first=True).as_matrix()
    return pose
