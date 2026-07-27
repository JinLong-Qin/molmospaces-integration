"""Bounded no-rollout smoke for the official sim_eval YAM MolmoSpaces bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from molmo_spaces.data_generation.config.molmoact2_official_sim_eval_yam_box_config import (
    OFFICIAL_SIM_EVAL_YAM_XML,
    MolmoAct2OfficialSimEvalYamBoxDataGenConfig,
)
from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
    build_molmoact2_yam_state,
    molmoact2_yam_action_to_move_group_command,
)
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.official_sim_eval_bimanual_yam_view import (
    OFFICIAL_YAM_GRIPPER_CLOSED_COMMAND,
    OFFICIAL_YAM_GRIPPER_OPEN_COMMAND,
)


def main() -> None:
    config = MolmoAct2OfficialSimEvalYamBoxDataGenConfig()
    robot_config = config.robot_config
    xml_path = robot_config.get_robot_xml_path()
    if xml_path != OFFICIAL_SIM_EVAL_YAM_XML:
        raise AssertionError(f"Unexpected XML path: {xml_path}")
    if not xml_path.exists():
        raise AssertionError(f"Missing official XML: {xml_path}")
    if robot_config.base_size is not None:
        raise AssertionError(f"Expected no MolmoSpaces base platform, got {robot_config.base_size}")

    spec = Robot._load_robot_spec(robot_config)
    required = {
        "bodies": ["bimanual_base", "left_link_6", "right_link_6"],
        "joints": [
            "left_joint1",
            "left_joint6",
            "left_left_finger",
            "right_joint1",
            "right_joint6",
            "right_right_finger",
        ],
        "actuators": [
            "left_joint1",
            "left_gripper_left_tip",
            "left_gripper_right_tip",
            "right_joint1",
            "right_gripper_left_tip",
            "right_gripper_right_tip",
        ],
        "sites": ["left_grasp_site", "right_grasp_site"],
    }
    missing = {kind: [] for kind in required}
    for name in required["bodies"]:
        if spec.body(name) is None:
            missing["bodies"].append(name)
    for name in required["joints"]:
        if spec.joint(name) is None:
            missing["joints"].append(name)
    for name in required["actuators"]:
        if spec.actuator(name) is None:
            missing["actuators"].append(name)
    for name in required["sites"]:
        if spec.site(name) is None:
            missing["sites"].append(name)
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise AssertionError(f"Official XML missing required names: {missing}")

    closed_state = build_molmoact2_yam_state(
        {
            "left_arm": np.zeros(6, dtype=np.float32),
            "right_arm": np.zeros(6, dtype=np.float32),
            "left_gripper": np.array([OFFICIAL_YAM_GRIPPER_CLOSED_COMMAND] * 2),
            "right_gripper": np.array([OFFICIAL_YAM_GRIPPER_CLOSED_COMMAND] * 2),
        },
        gripper_max=abs(OFFICIAL_YAM_GRIPPER_OPEN_COMMAND),
        gripper_open_command=OFFICIAL_YAM_GRIPPER_OPEN_COMMAND,
        gripper_closed_command=OFFICIAL_YAM_GRIPPER_CLOSED_COMMAND,
    )
    open_state = build_molmoact2_yam_state(
        {
            "left_arm": np.zeros(6, dtype=np.float32),
            "right_arm": np.zeros(6, dtype=np.float32),
            "left_gripper": np.array([OFFICIAL_YAM_GRIPPER_OPEN_COMMAND] * 2),
            "right_gripper": np.array([OFFICIAL_YAM_GRIPPER_OPEN_COMMAND] * 2),
        },
        gripper_max=abs(OFFICIAL_YAM_GRIPPER_OPEN_COMMAND),
        gripper_open_command=OFFICIAL_YAM_GRIPPER_OPEN_COMMAND,
        gripper_closed_command=OFFICIAL_YAM_GRIPPER_CLOSED_COMMAND,
    )
    if closed_state[6] != 0.0 or closed_state[13] != 0.0:
        raise AssertionError(f"Closed state should normalize to 0: {closed_state[[6, 13]]}")
    if open_state[6] != 1.0 or open_state[13] != 1.0:
        raise AssertionError(f"Open state should normalize to 1: {open_state[[6, 13]]}")

    command = molmoact2_yam_action_to_move_group_command(
        np.array([0.0] * 6 + [1.0] + [0.0] * 6 + [1.0], dtype=np.float32),
        gripper_max=abs(OFFICIAL_YAM_GRIPPER_OPEN_COMMAND),
        gripper_open_command=OFFICIAL_YAM_GRIPPER_OPEN_COMMAND,
        gripper_closed_command=OFFICIAL_YAM_GRIPPER_CLOSED_COMMAND,
    )
    if command["left_gripper"].shape != (1,) or command["right_gripper"].shape != (1,):
        raise AssertionError(f"Unexpected gripper command shape: {command}")
    if command["left_gripper"][0] != OFFICIAL_YAM_GRIPPER_OPEN_COMMAND:
        raise AssertionError(f"Open action did not map to official command: {command}")

    report = {
        "official_xml": str(xml_path),
        "bridge_config": config.__class__.__name__,
        "robot_view_factory": robot_config.robot_view_factory.__name__,
        "base_size": robot_config.base_size,
        "gripper_closed_state_fields": closed_state[[6, 13]].tolist(),
        "gripper_open_state_fields": open_state[[6, 13]].tolist(),
        "open_command": {
            "left_gripper": command["left_gripper"].tolist(),
            "right_gripper": command["right_gripper"].tolist(),
        },
        "evidence_level": "transport/import/xml-contract/policy-scale smoke only; no env rollout",
    }
    output = Path(
        "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_smoke_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
