"""No-rollout smoke checks for the official MolmoAct2 YAM box preset."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/molmospaces-matplotlib")
os.environ.setdefault("NLTK_DATA", "/tmp/molmospaces-nltk-data")

import numpy as np

import molmo_spaces.data_generation.config.molmoact2_official_yam_box_config as official_config
from molmo_spaces.data_generation.config.molmoact2_official_yam_box_config import (
    OFFICIAL_MOLMOACT2_YAM_ENV_ID,
    OFFICIAL_MOLMOACT2_YAM_INSTRUCTION,
    OFFICIAL_YAM_BALL_NAME,
    OFFICIAL_YAM_BOX_INNER_HALF,
    OFFICIAL_YAM_BOX_POS_XY,
    OFFICIAL_YAM_BOX_RIM_Z,
    OFFICIAL_YAM_BOX_WALL,
    OFFICIAL_YAM_CAMERA_SPECS,
    OFFICIAL_YAM_LEGO_NAME,
    OFFICIAL_YAM_OBJECT_ANCHORS_XY,
    OFFICIAL_YAM_OPEN_BOX_NAME,
    OFFICIAL_YAM_SPAWN_NOISE,
    MolmoAct2OfficialYamBoxCameraSystem,
    MolmoAct2OfficialYamBoxDataGenConfig,
    MolmoAct2OfficialYamBoxTaskSampler,
    official_yam_box_success_from_positions,
)


class FakeBody:
    def __init__(self, name: str, position: tuple[float, float, float]) -> None:
        self.name = name
        self.pose_set_count = 0
        self.pose = np.eye(4, dtype=float)
        self.pose[:3, 3] = np.asarray(position, dtype=float)
        self.pose_set_count = 0

    @property
    def pose(self) -> np.ndarray:
        return self._pose

    @pose.setter
    def pose(self, value: np.ndarray) -> None:
        self._pose = np.asarray(value, dtype=float)
        self.pose_set_count += 1

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3, 3]


class FakeMoveGroup:
    def __init__(self) -> None:
        self.joint_pos = None


class FakeRobotBase:
    def __init__(self) -> None:
        self.pose = None


class FakeRobotView:
    def __init__(self) -> None:
        self.base = FakeRobotBase()
        self._move_groups = {
            "left_arm": FakeMoveGroup(),
            "right_arm": FakeMoveGroup(),
            "left_gripper": FakeMoveGroup(),
            "right_gripper": FakeMoveGroup(),
        }

    def move_group_ids(self) -> list[str]:
        return list(self._move_groups)

    def get_move_group(self, group_name: str) -> FakeMoveGroup:
        return self._move_groups[group_name]


class FakeEnv:
    def __init__(self) -> None:
        self.current_model = object()
        self.current_data = object()
        self.bodies = {
            OFFICIAL_YAM_LEGO_NAME: FakeBody(OFFICIAL_YAM_LEGO_NAME, (-0.30, 0.22, 0.770)),
            OFFICIAL_YAM_BALL_NAME: FakeBody(OFFICIAL_YAM_BALL_NAME, (-0.30, -0.22, 0.773)),
            OFFICIAL_YAM_OPEN_BOX_NAME: FakeBody(OFFICIAL_YAM_OPEN_BOX_NAME, (-0.15, 0.0, 0.745)),
        }


def test_config_contract_matches_official_source_constants() -> None:
    config = MolmoAct2OfficialYamBoxDataGenConfig()

    assert OFFICIAL_MOLMOACT2_YAM_ENV_ID == "BimanualYAMPutEverythingInBox-v1"
    assert OFFICIAL_MOLMOACT2_YAM_INSTRUCTION == "put everything into the box"
    assert OFFICIAL_YAM_OBJECT_ANCHORS_XY == {
        OFFICIAL_YAM_LEGO_NAME: (-0.30, 0.22),
        OFFICIAL_YAM_BALL_NAME: (-0.30, -0.22),
    }
    assert OFFICIAL_YAM_BOX_POS_XY == (-0.15, 0.0)
    np.testing.assert_allclose(
        official_config.OFFICIAL_YAM_ROBOT_WORLD_POS,
        np.array([-0.65, 0.0, 0.01]),
    )
    assert config.task_horizon == 400
    assert config.policy_config.instruction_override == OFFICIAL_MOLMOACT2_YAM_INSTRUCTION
    assert config.policy_config.camera_mapping == {
        "top_cam": "top_cam",
        "left_cam": "left_cam",
        "right_cam": "right_cam",
    }
    assert config.task_sampler_config.scene_xml_paths == [
        "examples/molmoact2_official_yam_box/scene.xml"
    ]
    assert config.task_sampler_config.sim_settle_timesteps == 0
    assert config.robot_config.init_qpos["left_arm"] == [0.0] * 6
    assert config.robot_config.init_qpos["right_arm"] == [0.0] * 6
    assert config.robot_config.init_qpos["left_gripper"] == [0.041, 0.0]
    assert config.robot_config.init_qpos["right_gripper"] == [0.041, 0.0]


def test_task_sensor_contract_includes_behavior_fields() -> None:
    config = MolmoAct2OfficialYamBoxDataGenConfig()
    task = object.__new__(official_config.MolmoAct2OfficialYamBoxTask)
    sensor_suite = task._create_sensor_suite_from_config(config)

    assert "task_info" in sensor_suite.sensors
    assert f"grasp_state_{OFFICIAL_YAM_LEGO_NAME}" in sensor_suite.sensors
    assert f"grasp_state_{OFFICIAL_YAM_BALL_NAME}" in sensor_suite.sensors
    assert f"{OFFICIAL_YAM_LEGO_NAME}_start_pose" in sensor_suite.sensors
    assert f"{OFFICIAL_YAM_BALL_NAME}_start_pose" in sensor_suite.sensors
    assert sensor_suite.sensors["task_info"].str_max_len == 12000


def test_camera_contract_matches_official_robot_source() -> None:
    camera_system = MolmoAct2OfficialYamBoxCameraSystem()
    assert camera_system.img_resolution == (640, 360)

    cameras = {camera.name: camera for camera in camera_system.cameras}
    assert set(cameras) == {"top_cam", "left_cam", "right_cam"}
    for name, spec in OFFICIAL_YAM_CAMERA_SPECS.items():
        camera = cameras[name]
        assert camera.camera_offset == spec["p"]
        assert spec["q"] != spec["molmospaces_q"]
        assert camera.camera_quaternion == spec["molmospaces_q"]
        assert camera.fov == spec["hfov_deg"]
        assert spec["resolution"] == (640, 360)

    np.testing.assert_allclose(
        cameras["top_cam"].camera_quaternion,
        [
            -0.7044160264027587,
            -0.061628416716219554,
            0.061628416716219554,
            0.7044160264027587,
        ],
    )
    np.testing.assert_allclose(
        cameras["left_cam"].camera_quaternion,
        [
            7.038474420485891e-09,
            1.8859535483010483e-09,
            0.9659258277806714,
            0.2588190395357825,
        ],
    )
    np.testing.assert_allclose(
        cameras["right_cam"].camera_quaternion,
        [
            7.038474420485891e-09,
            1.8859535483010483e-09,
            0.9659258277806714,
            0.2588190395357825,
        ],
    )
    assert cameras["top_cam"].reference_body_names == ["robot_0/bimanual_base", "robot_0/base"]
    assert cameras["left_cam"].reference_body_names == ["robot_0/left_link_6"]
    assert cameras["right_cam"].reference_body_names == ["robot_0/right_link_6"]
    assert cameras["top_cam"].visibility_constraints == {"__official_yam_task_objects__": 0.0001}


def test_success_predicate_matches_official_center_in_box_rule() -> None:
    success = official_yam_box_success_from_positions(
        {
            OFFICIAL_YAM_LEGO_NAME: np.array(
                [OFFICIAL_YAM_BOX_POS_XY[0], OFFICIAL_YAM_BOX_POS_XY[1], OFFICIAL_YAM_BOX_RIM_Z]
            ),
            OFFICIAL_YAM_BALL_NAME: np.array(
                [
                    OFFICIAL_YAM_BOX_POS_XY[0] + OFFICIAL_YAM_BOX_INNER_HALF * 0.5,
                    OFFICIAL_YAM_BOX_POS_XY[1],
                    OFFICIAL_YAM_BOX_WALL + 0.745,
                ]
            ),
        }
    )
    assert success["success"] is True
    assert success["n_in_box"] == 2

    failure = official_yam_box_success_from_positions(
        {
            OFFICIAL_YAM_LEGO_NAME: np.array([-0.30, 0.22, 0.770]),
            OFFICIAL_YAM_BALL_NAME: np.array([-0.30, -0.22, 0.773]),
        }
    )
    assert failure["success"] is False
    assert failure["n_in_box"] == 0


def test_sampler_applies_sourced_spawn_noise_within_official_bounds(monkeypatch) -> None:
    env = FakeEnv()
    robot_view = FakeRobotView()
    config = MolmoAct2OfficialYamBoxDataGenConfig()
    sampler = MolmoAct2OfficialYamBoxTaskSampler(config)

    monkeypatch.setattr(
        official_config,
        "create_mlspaces_body",
        lambda _data, name: env.bodies[name],
    )
    monkeypatch.setattr(official_config.mujoco, "mj_resetData", lambda _model, _data: None)
    monkeypatch.setattr(official_config.mujoco, "mj_forward", lambda _model, _data: None)
    monkeypatch.setattr(
        official_config.np.random,
        "uniform",
        lambda low, high, size=None: np.array([0.01, -0.01]) if size == 2 else 0.0,
    )

    sampler.randomize_scene(env, robot_view)

    expected_offsets = {
        OFFICIAL_YAM_LEGO_NAME: np.array([0.01, -0.01]),
        OFFICIAL_YAM_BALL_NAME: np.array([0.01, -0.01]),
    }
    for name, anchor in OFFICIAL_YAM_OBJECT_ANCHORS_XY.items():
        position_xy = env.bodies[name].position[:2]
        offset = position_xy - np.asarray(anchor)
        np.testing.assert_allclose(offset, expected_offsets[name])
        assert np.all(np.abs(offset) <= OFFICIAL_YAM_SPAWN_NOISE)

    assert env.bodies[OFFICIAL_YAM_OPEN_BOX_NAME].position.tolist() == [-0.15, 0.0, 0.745]
    assert env.bodies[OFFICIAL_YAM_OPEN_BOX_NAME].pose_set_count == 0
    np.testing.assert_allclose(robot_view.base.pose[:3, 3], np.array([-0.65, 0.0, 0.01]))


def main() -> None:
    print("running config contract", flush=True)
    test_config_contract_matches_official_source_constants()
    print("running sensor contract", flush=True)
    test_task_sensor_contract_includes_behavior_fields()
    print("running camera contract", flush=True)
    test_camera_contract_matches_official_robot_source()
    print("running success predicate", flush=True)
    test_success_predicate_matches_official_center_in_box_rule()

    class MonkeyPatch:
        def __init__(self) -> None:
            self._patches = []

        def setattr(self, target, name, value=None) -> None:
            if value is None:
                dotted_name = target
                import importlib

                module_name, attr_name = dotted_name.rsplit(".", 1)
                target = importlib.import_module(module_name)
                value = name
                name = attr_name

            old_value = getattr(target, name)
            setattr(target, name, value)
            self._patches.append((target, name, old_value))

        def undo(self) -> None:
            for module, attr_name, old_value in reversed(self._patches):
                setattr(module, attr_name, old_value)

    monkeypatch = MonkeyPatch()
    try:
        print("running sampler contract", flush=True)
        test_sampler_applies_sourced_spawn_noise_within_official_bounds(monkeypatch)
    finally:
        monkeypatch.undo()
    print("molmoact2_official_yam_box smoke passed")


if __name__ == "__main__":
    main()
