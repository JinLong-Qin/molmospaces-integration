from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from molmo_spaces.data_generation.config.molmoact2_official_sim_eval_yam_box_config import (
    MolmoAct2OfficialSimEvalYamBoxDataGenConfig,
)
from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
    execute_molmoact2_yam_action,
    molmoact2_yam_action_to_move_group_command,
)


def as_list(x):
    return np.asarray(x, dtype=float).reshape(-1).tolist()


def main():
    cfg = MolmoAct2OfficialSimEvalYamBoxDataGenConfig()
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    try:
        task = sampler.sample_task(force_advance_scene=True, house_index=0)
        rv = task._env.current_robot.robot_view
        before = {
            "left_gripper": as_list(rv.get_move_group("left_gripper").joint_pos),
            "right_gripper": as_list(rv.get_move_group("right_gripper").joint_pos),
        }
        action14 = np.zeros(14, dtype=np.float32)
        action14[6] = 1.0
        action14[13] = 1.0
        command = molmoact2_yam_action_to_move_group_command(
            action14,
            gripper_max=cfg.policy_config.gripper_max,
            gripper_open_command=cfg.policy_config.gripper_open_command,
            gripper_closed_command=cfg.policy_config.gripper_closed_command,
            grasping_type=cfg.policy_config.grasping_type,
            grasping_threshold=cfg.policy_config.grasping_threshold,
        )
        obs, reward, terminated, truncated, info = execute_molmoact2_yam_action(
            task,
            command,
            execution_mode=cfg.policy_config.execution_mode,
            joint_step=0.02,
            max_smoothing_steps=3,
            command_hz=cfg.policy_config.execution_command_hz,
        )
        after = {
            "left_gripper": as_list(rv.get_move_group("left_gripper").joint_pos),
            "right_gripper": as_list(rv.get_move_group("right_gripper").joint_pos),
        }
        report = {
            "evidence_level": "one commanded robot-step smoke using local 14D action conversion; no /act and no rollout",
            "command": {k: as_list(v) for k, v in command.items()},
            "before_gripper_qpos": before,
            "after_gripper_qpos": after,
            "reward": as_list(reward),
            "terminated": np.asarray(terminated).astype(bool).tolist(),
            "truncated": np.asarray(truncated).astype(bool).tolist(),
            "success_info": info,
            "execution_mode": cfg.policy_config.execution_mode,
            "episode_step_count": int(task.episode_step_count),
        }
        out = Path(
            "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_step_smoke_report.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
