"""Official MolmoAct2 YAM put-everything-in-box preset for MolmoSpaces.

This preset mirrors the public MolmoAct2 ManiSkill sim_eval contract while
remaining a lightweight MolmoSpaces/MuJoCo approximation:

- env ID: BimanualYAMPutEverythingInBox-v1
- instruction: put everything into the box
- objects: 073-a_lego_duplo and 056_tennis_ball
- cameras: top_cam, left_cam, right_cam at 640x360
- state/action wire shape: 14

The YCB objects are represented as MuJoCo primitives to avoid large downloads.
The task success predicate follows the official center-in-box XY/Z rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.base_packing_configs import PackingDataGenConfig
from molmo_spaces.configs.camera_configs import CameraSystemConfig, RobotMountedCameraConfig
from molmo_spaces.configs.policy_configs_baselines import MolmoAct2YamPolicyConfig
from molmo_spaces.configs.robot_configs import BimanualYamRobotConfig
from molmo_spaces.configs.task_configs import PackingTaskConfig
from molmo_spaces.configs.task_sampler_configs import PackingTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.sensors import (
    GraspStateSensor,
    ObjectStartPoseSensor,
    TaskInfoSensor,
    get_core_sensors,
)
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms
from molmo_spaces.utils.pose import pose_mat_to_7d, pos_quat_to_pose_mat

OFFICIAL_MOLMOACT2_YAM_ENV_ID = "BimanualYAMPutEverythingInBox-v1"
OFFICIAL_MOLMOACT2_YAM_INSTRUCTION = "put everything into the box"
OFFICIAL_MOLMOACT2_YAM_SOURCE_TASK = "sim_eval/tasks/yam_tasks/bimanual_put_everything_in_box.py"
OFFICIAL_MOLMOACT2_YAM_SOURCE_ROBOT = "sim_eval/robots/bimanual_yam.py"

OFFICIAL_YAM_LEGO_NAME = "obj_073-a_lego_duplo"
OFFICIAL_YAM_BALL_NAME = "obj_056_tennis_ball"
OFFICIAL_YAM_OBJECT_NAMES = (OFFICIAL_YAM_LEGO_NAME, OFFICIAL_YAM_BALL_NAME)
OFFICIAL_YAM_OPEN_BOX_NAME = "open_box"

OFFICIAL_YAM_OBJECT_ANCHORS_XY: dict[str, tuple[float, float]] = {
    OFFICIAL_YAM_LEGO_NAME: (-0.30, 0.22),
    OFFICIAL_YAM_BALL_NAME: (-0.30, -0.22),
}
OFFICIAL_YAM_SPAWN_NOISE = 0.02
OFFICIAL_YAM_BOX_POS_XY = (-0.15, 0.0)
OFFICIAL_YAM_BOX_INNER_HALF = 0.09
OFFICIAL_YAM_BOX_HEIGHT = 0.06
OFFICIAL_YAM_BOX_WALL = 0.008
OFFICIAL_YAM_BOX_WORLD_Z = 0.745000
OFFICIAL_YAM_BOX_FLOOR_TOP_Z = OFFICIAL_YAM_BOX_WORLD_Z + OFFICIAL_YAM_BOX_WALL
OFFICIAL_YAM_BOX_RIM_Z = OFFICIAL_YAM_BOX_FLOOR_TOP_Z + OFFICIAL_YAM_BOX_HEIGHT
OFFICIAL_YAM_ROBOT_WORLD_POS = (-0.65, 0.0, 0.01)

OFFICIAL_YAM_CAMERA_SPECS: dict[str, dict[str, Any]] = {
    "top_cam": {
        "mount": "bimanual_base",
        "resolution": (640, 360),
        "hfov_deg": 69.4,
        "p": [0.15, 0.0, 0.8],
        "q": [0.7660444431189782, 0.0, 0.6427876096865391, 0.0],
    },
    "left_cam": {
        "mount": "left_link_6",
        "resolution": (640, 360),
        "hfov_deg": 87.0,
        "p": [0.0, 0.09, 0.06],
        "q": [
            0.612372429196013,
            -0.35355339154618404,
            -0.3535533966987049,
            -0.612372438120441,
        ],
    },
    "right_cam": {
        "mount": "right_link_6",
        "resolution": (640, 360),
        "hfov_deg": 87.0,
        "p": [0.0, 0.09, 0.06],
        "q": [
            0.612372429196013,
            -0.35355339154618404,
            -0.3535533966987049,
            -0.612372438120441,
        ],
    },
}


def _sapien_camera_quat_to_molmospaces_camera_quat(quat_wxyz: list[float]) -> list[float]:
    """Convert official SAPIEN camera pose orientation to MolmoSpaces convention.

    The official ManiSkill/SAPIEN camera pose uses local +X as optical forward
    and local +Z as up. MolmoSpaces camera manager uses local -Z as optical
    forward and local +Y as up. This keeps the sourced official camera frame
    direction while expressing it in MolmoSpaces' local camera axes.
    """

    sapien_rot = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    sapien_to_molmospaces_camera_axes = np.column_stack(
        (
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        )
    )
    molmospaces_rot = sapien_rot @ sapien_to_molmospaces_camera_axes
    return R.from_matrix(molmospaces_rot).as_quat(scalar_first=True).tolist()


for _camera_spec in OFFICIAL_YAM_CAMERA_SPECS.values():
    _camera_spec["molmospaces_q"] = _sapien_camera_quat_to_molmospaces_camera_quat(
        _camera_spec["q"]
    )


def official_yam_box_success_from_positions(
    object_positions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate the official YAM in-box predicate on object center positions."""

    bx, by = OFFICIAL_YAM_BOX_POS_XY
    per_object: dict[str, dict[str, Any]] = {}
    n_in_box = 0
    for name in OFFICIAL_YAM_OBJECT_NAMES:
        pos = np.asarray(object_positions[name], dtype=float).reshape(3)
        dx = abs(float(pos[0]) - bx)
        dy = abs(float(pos[1]) - by)
        inside_xy = dx < OFFICIAL_YAM_BOX_INNER_HALF and dy < OFFICIAL_YAM_BOX_INNER_HALF
        z_ok = (
            pos[2] > OFFICIAL_YAM_BOX_FLOOR_TOP_Z - 0.01 and pos[2] < OFFICIAL_YAM_BOX_RIM_Z + 0.05
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


def _pose7d_to_mat(xyz_quat: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    pose = np.asarray(xyz_quat, dtype=float)
    return pos_quat_to_pose_mat(pose[:3], pose[3:7])


def _identity_pose_at(pos: tuple[float, float, float]) -> np.ndarray:
    return _pose7d_to_mat([pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0])


def _sample_official_yam_object_pose(
    anchor_xy: tuple[float, float],
    z: float,
    *,
    spawn_noise: float = OFFICIAL_YAM_SPAWN_NOISE,
) -> np.ndarray:
    """Sample MolmoSpaces object pose with official YAM XY noise semantics."""

    xy_noise = np.random.uniform(-spawn_noise, spawn_noise, size=2)
    yaw = float(np.random.uniform(-np.pi, np.pi))
    quat = R.from_euler("z", yaw).as_quat(scalar_first=True)
    return pos_quat_to_pose_mat(
        [anchor_xy[0] + xy_noise[0], anchor_xy[1] + xy_noise[1], z],
        quat,
    )


class MolmoAct2OfficialYamBoxCameraSystem(CameraSystemConfig):
    """Official MolmoAct2 YAM camera triad in MolmoSpaces camera config form."""

    img_resolution: tuple[int, int] = (640, 360)
    cameras: list[RobotMountedCameraConfig] = [
        RobotMountedCameraConfig(
            name="top_cam",
            reference_body_names=["robot_0/bimanual_base", "robot_0/base"],
            camera_offset=OFFICIAL_YAM_CAMERA_SPECS["top_cam"]["p"],
            camera_quaternion=OFFICIAL_YAM_CAMERA_SPECS["top_cam"]["molmospaces_q"],
            fov=OFFICIAL_YAM_CAMERA_SPECS["top_cam"]["hfov_deg"],
            visibility_constraints={"__official_yam_task_objects__": 0.0001},
        ),
        RobotMountedCameraConfig(
            name="left_cam",
            reference_body_names=["robot_0/left_link_6"],
            camera_offset=OFFICIAL_YAM_CAMERA_SPECS["left_cam"]["p"],
            camera_quaternion=OFFICIAL_YAM_CAMERA_SPECS["left_cam"]["molmospaces_q"],
            fov=OFFICIAL_YAM_CAMERA_SPECS["left_cam"]["hfov_deg"],
        ),
        RobotMountedCameraConfig(
            name="right_cam",
            reference_body_names=["robot_0/right_link_6"],
            camera_offset=OFFICIAL_YAM_CAMERA_SPECS["right_cam"]["p"],
            camera_quaternion=OFFICIAL_YAM_CAMERA_SPECS["right_cam"]["molmospaces_q"],
            fov=OFFICIAL_YAM_CAMERA_SPECS["right_cam"]["hfov_deg"],
        ),
    ]


class MolmoAct2OfficialYamBoxTask(BaseMujocoTask):
    """Two-object official-style YAM box task metric."""

    def get_task_description(self) -> str:
        return OFFICIAL_MOLMOACT2_YAM_INSTRUCTION

    def get_task_objects(self, batch_index: int = 0) -> dict[str, str]:
        task_objects = super().get_task_objects(batch_index)
        task_objects.update(
            {
                "lego_duplo": OFFICIAL_YAM_LEGO_NAME,
                "tennis_ball": OFFICIAL_YAM_BALL_NAME,
                "open_box": OFFICIAL_YAM_OPEN_BOX_NAME,
            }
        )
        return task_objects

    def _create_sensor_suite_from_config(self, config) -> SensorSuite:
        sensors = [
            TaskInfoSensor(uuid="task_info", str_max_len=12000)
            if sensor.uuid == "task_info"
            else sensor
            for sensor in get_core_sensors(config)
        ]
        for object_name in OFFICIAL_YAM_OBJECT_NAMES:
            sensors.extend(
                [
                    ObjectStartPoseSensor(
                        object_name=object_name, uuid=f"{object_name}_start_pose"
                    ),
                    GraspStateSensor(object_name=object_name, uuid=f"grasp_state_{object_name}"),
                ]
            )
        return SensorSuite(sensors)

    def _object_positions(self, batch_index: int = 0) -> dict[str, np.ndarray]:
        data = self._env.mj_datas[batch_index]
        return {
            name: np.asarray(create_mlspaces_body(data, name).position, dtype=float)
            for name in OFFICIAL_YAM_OBJECT_NAMES
        }

    def _body_pose_summary(self, batch_index: int = 0) -> dict[str, dict[str, list[float]]]:
        data = self._env.mj_datas[batch_index]
        summaries: dict[str, dict[str, list[float]]] = {}
        for name in OFFICIAL_YAM_OBJECT_NAMES + (OFFICIAL_YAM_OPEN_BOX_NAME,):
            body = create_mlspaces_body(data, name)
            summaries[name] = {
                "position": np.asarray(body.position, dtype=float).tolist(),
                "quaternion": np.asarray(body.quat, dtype=float).tolist(),
            }
        return summaries

    def _gripper_pose_summary(self, batch_index: int = 0) -> dict[str, dict[str, list[float]]]:
        robot_view = self._env.robots[batch_index].robot_view
        summaries: dict[str, dict[str, list[float]]] = {}
        for gripper_name in ("left_gripper", "right_gripper"):
            if gripper_name not in robot_view.move_group_ids():
                continue
            pose = pose_mat_to_7d(robot_view.get_move_group(gripper_name).leaf_frame_to_world)
            summaries[gripper_name] = {
                "position": np.asarray(pose[:3], dtype=float).tolist(),
                "quaternion": np.asarray(pose[3:7], dtype=float).tolist(),
            }
        return summaries

    def _gripper_object_distances(
        self,
        object_poses: dict[str, dict[str, list[float]]],
        gripper_poses: dict[str, dict[str, list[float]]],
    ) -> dict[str, dict[str, float]]:
        distances: dict[str, dict[str, float]] = {}
        for gripper_name, gripper_pose in gripper_poses.items():
            gripper_pos = np.asarray(gripper_pose["position"], dtype=float)
            distances[gripper_name] = {}
            for object_name in OFFICIAL_YAM_OBJECT_NAMES:
                object_pos = np.asarray(object_poses[object_name]["position"], dtype=float)
                distances[gripper_name][object_name] = float(
                    np.linalg.norm(gripper_pos - object_pos)
                )
        return distances

    def _contact_grasp_summary(self, batch_index: int = 0) -> dict[str, Any]:
        model = self._env.mj_model
        data = self._env.mj_datas[batch_index]
        robot_view = self._env.robots[batch_index].robot_view

        gripper_geoms: dict[str, set[int]] = {}
        for gripper_name in ("left_gripper", "right_gripper"):
            if gripper_name not in robot_view.move_group_ids():
                continue
            gripper_geoms[gripper_name] = descendant_geoms(
                model,
                robot_view.get_move_group(gripper_name).root_body_id,
                visible_only=False,
            )

        contact_summary: dict[str, Any] = {}
        for object_name in OFFICIAL_YAM_OBJECT_NAMES:
            object_body = create_mlspaces_body(data, object_name)
            object_geoms = set(descendant_geoms(model, object_body.body_id, visible_only=False))
            touching = {gripper_name: False for gripper_name in gripper_geoms}
            held = True
            contact_count = 0
            nongripper_contact_count = 0

            for contact_idx in range(data.ncon):
                contact = data.contact[contact_idx]
                object_in_geom1 = contact.geom1 in object_geoms
                object_in_geom2 = contact.geom2 in object_geoms
                if object_in_geom1 == object_in_geom2:
                    continue

                contact_count += 1
                other_geom = contact.geom2 if object_in_geom1 else contact.geom1
                for gripper_name, geom_ids in gripper_geoms.items():
                    if other_geom in geom_ids:
                        touching[gripper_name] = True
                        break
                else:
                    nongripper_contact_count += 1
                    held = False

            per_gripper = {
                gripper_name: {
                    "touching": bool(is_touching),
                    "held": bool(held and is_touching),
                }
                for gripper_name, is_touching in touching.items()
            }
            contact_summary[object_name] = {
                "contact_count": int(contact_count),
                "nongripper_contact_count": int(nongripper_contact_count),
                "any_gripper_touching": bool(any(touching.values())),
                "per_gripper": per_gripper,
            }
        return contact_summary

    def evaluate_official_yam_success(self, batch_index: int = 0) -> dict[str, Any]:
        return official_yam_box_success_from_positions(self._object_positions(batch_index))

    def judge_success(self) -> bool:
        return bool(self.evaluate_official_yam_success(0)["success"])

    def get_reward(self) -> np.ndarray:
        rewards = np.zeros(self._env.n_batch)
        for i in range(self._env.n_batch):
            metrics = self.evaluate_official_yam_success(i)
            rewards[i] = metrics["n_in_box"] / metrics["n_total"]
        return rewards

    def get_info(self) -> list[dict[str, Any]]:
        infos = []
        for i in range(self._env.n_batch):
            metrics = self.evaluate_official_yam_success(i)
            object_poses = self._body_pose_summary(i)
            gripper_poses = self._gripper_pose_summary(i)
            infos.append(
                {
                    "success": metrics["success"],
                    "n_in_box": metrics["n_in_box"],
                    "n_total": metrics["n_total"],
                    "official_yam_box": metrics,
                    "official_yam_behavior": {
                        "object_poses": object_poses,
                        "gripper_poses": gripper_poses,
                        "gripper_object_distances": self._gripper_object_distances(
                            object_poses, gripper_poses
                        ),
                        "contacts": self._contact_grasp_summary(i),
                    },
                    "episode_step": self.episode_step_count,
                }
            )
        return infos

    def get_obs_scene(self):
        obs_scene = super().get_obs_scene()
        obs_scene.update(
            {
                "text": OFFICIAL_MOLMOACT2_YAM_INSTRUCTION,
                "object_names": list(OFFICIAL_YAM_OBJECT_NAMES),
                "place_receptacle_name": OFFICIAL_YAM_OPEN_BOX_NAME,
                "official_env_id": OFFICIAL_MOLMOACT2_YAM_ENV_ID,
            }
        )
        return obs_scene


class MolmoAct2OfficialYamBoxTaskSampler(BaseMujocoTaskSampler):
    """Deterministic fixed-scene sampler for the official YAM box task."""

    def add_auxiliary_objects(self, spec) -> None:
        self.config.policy_config.policy_cls.add_auxiliary_objects(self.config, spec)

    def randomize_scene(self, env, robot_view) -> None:
        super().randomize_scene(env, robot_view)
        model, data = env.current_model, env.current_data
        mujoco.mj_resetData(model, data)

        for group_name, qpos in self.config.robot_config.init_qpos.items():
            if group_name in robot_view.move_group_ids():
                robot_view.get_move_group(group_name).joint_pos = np.asarray(qpos, dtype=float)

        robot_view.base.pose = _identity_pose_at(OFFICIAL_YAM_ROBOT_WORLD_POS)

        for name, anchor_xy in OFFICIAL_YAM_OBJECT_ANCHORS_XY.items():
            obj = create_mlspaces_body(data, name)
            obj.pose = _sample_official_yam_object_pose(anchor_xy, float(obj.position[2]))

        mujoco.mj_forward(model, data)

    def get_workspace_center(self, env) -> np.ndarray:
        points = [
            np.array([*xy, OFFICIAL_YAM_BOX_FLOOR_TOP_Z], dtype=float)
            for xy in OFFICIAL_YAM_OBJECT_ANCHORS_XY.values()
        ]
        points.append(np.array([*OFFICIAL_YAM_BOX_POS_XY, OFFICIAL_YAM_BOX_RIM_Z], dtype=float))
        return np.mean(points, axis=0)

    def resolve_visibility_object(self, env, key: str) -> list[str]:
        if key in ("__task_objects__", "__official_yam_task_objects__"):
            return [*OFFICIAL_YAM_OBJECT_NAMES, OFFICIAL_YAM_OPEN_BOX_NAME]
        return super().resolve_visibility_object(env, key)

    def _sample_task(self, env) -> MolmoAct2OfficialYamBoxTask:
        task_cfg = self.config.task_config
        task_cfg.pickup_obj_name = OFFICIAL_YAM_LEGO_NAME
        task_cfg.place_receptacle_name = OFFICIAL_YAM_OPEN_BOX_NAME
        task_cfg.referral_expressions = {
            "pickup_name": "everything",
            "place_name": "box",
            "instruction": OFFICIAL_MOLMOACT2_YAM_INSTRUCTION,
        }
        task_cfg.added_objects = {
            OFFICIAL_YAM_LEGO_NAME: Path("examples/molmoact2_official_yam_box/scene.xml"),
            OFFICIAL_YAM_BALL_NAME: Path("examples/molmoact2_official_yam_box/scene.xml"),
            OFFICIAL_YAM_OPEN_BOX_NAME: Path("examples/molmoact2_official_yam_box/scene.xml"),
        }

        data = env.current_data
        task_cfg.object_poses = {
            name: pose_mat_to_7d(create_mlspaces_body(data, name).pose).tolist()
            for name in OFFICIAL_YAM_OBJECT_NAMES + (OFFICIAL_YAM_OPEN_BOX_NAME,)
        }
        task_cfg.pickup_obj_start_pose = task_cfg.object_poses[OFFICIAL_YAM_LEGO_NAME]
        task_cfg.place_receptacle_start_pose = task_cfg.object_poses[OFFICIAL_YAM_OPEN_BOX_NAME]
        task_cfg.robot_base_pose = pose_mat_to_7d(env.current_robot.robot_view.base.pose).tolist()

        self.setup_cameras(env, deterministic_only=False)
        return MolmoAct2OfficialYamBoxTask(env, self.config)


@register_config("MolmoAct2OfficialYamBoxDataGenConfig")
class MolmoAct2OfficialYamBoxDataGenConfig(PackingDataGenConfig):
    """MolmoAct2 official YAM put-everything-in-box preset."""

    scene_dataset: str = "user"
    data_split: str = "train"
    seed: int = 20260622
    num_workers: int = 1
    task_horizon: int = 400
    policy_dt_ms: float = 40.0
    filter_for_successful_trajectories: bool = False
    output_dir: Path = Path("experiment_output")
    environment_light_intensity: float = 15000.0

    robot_config: BimanualYamRobotConfig = BimanualYamRobotConfig(
        init_qpos={
            "left_arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "right_arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "left_gripper": [0.041, 0.0],
            "right_gripper": [0.041, 0.0],
        },
        init_qpos_noise_range=None,
    )
    camera_config: MolmoAct2OfficialYamBoxCameraSystem = MolmoAct2OfficialYamBoxCameraSystem()
    task_sampler_config: PackingTaskSamplerConfig = PackingTaskSamplerConfig(
        task_sampler_class=MolmoAct2OfficialYamBoxTaskSampler,
        dataset_name="user",
        scene_xml_paths=["examples/molmoact2_official_yam_box/scene.xml"],
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
    task_config: PackingTaskConfig = PackingTaskConfig(task_cls=MolmoAct2OfficialYamBoxTask)
    policy_config: MolmoAct2YamPolicyConfig = MolmoAct2YamPolicyConfig(
        instruction_override=OFFICIAL_MOLMOACT2_YAM_INSTRUCTION,
        num_steps=30,
        n_action_steps=None,
        debug_dump_dir="artifacts/molmospaces/molmoact2_official_yam_box_debug",
        debug_dump_max_calls=2,
        raw_action_log_path=(
            "artifacts/molmospaces/molmoact2_official_yam_box_debug/raw_actions.jsonl"
        ),
        execution_command_hz=30.0,
        camera_mapping={
            "top_cam": "top_cam",
            "left_cam": "left_cam",
            "right_cam": "right_cam",
        },
    )

    @property
    def tag(self) -> str:
        return "molmoact2_official_yam_box"
