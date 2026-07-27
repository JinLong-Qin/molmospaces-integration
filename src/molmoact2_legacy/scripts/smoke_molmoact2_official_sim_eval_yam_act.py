from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np

from molmo_spaces.data_generation.config.molmoact2_official_sim_eval_yam_box_config import (
    MolmoAct2OfficialSimEvalYamBoxDataGenConfig,
)
from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
    MolmoAct2YamPolicy,
    execute_molmoact2_yam_action,
)


def arr_summary(x):
    a = np.asarray(x)
    out = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if a.size and np.issubdtype(a.dtype, np.number):
        out.update(
            {"min": float(np.nanmin(a)), "max": float(np.nanmax(a)), "mean": float(np.nanmean(a))}
        )
    return out


def as_list(x):
    return np.asarray(x, dtype=float).reshape(-1).tolist()


def main():
    cfg = MolmoAct2OfficialSimEvalYamBoxDataGenConfig()
    cfg.policy_config.num_steps = 1
    cfg.policy_config.n_action_steps = 1
    cfg.policy_config.timeout = 120.0
    cfg.policy_config.raw_action_log_path = "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_act_raw_actions.jsonl"
    cfg.policy_config.debug_dump_dir = (
        "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_act_debug"
    )
    cfg.policy_config.debug_dump_max_calls = 1
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    try:
        task = sampler.sample_task(force_advance_scene=True, house_index=0)
        policy = MolmoAct2YamPolicy(cfg, task=task)
        obs = task.get_observations()
        request = policy.obs_to_model_input(obs)
        started = time.time()
        raw_action = policy.inference_model(request)
        elapsed = time.time() - started
        command = policy.model_output_to_action(raw_action)
        observation, reward, terminated, truncated, info = execute_molmoact2_yam_action(
            task,
            command,
            joint_step=0.02,
            max_smoothing_steps=3,
            command_hz=cfg.policy_config.execution_command_hz,
        )
        rv = task._env.current_robot.robot_view
        report = {
            "evidence_level": "single remote /act request + first selected action executed with bounded smoothing; no rollout",
            "endpoint_url": policy.endpoint_url,
            "request_num_steps": request.get("num_steps"),
            "request_state": as_list(request["state"]),
            "request_image_summaries": {
                k: arr_summary(request[k]) for k in ("top_cam", "left_cam", "right_cam")
            },
            "inference_elapsed_sec": elapsed,
            "raw_action_shape": list(np.asarray(raw_action).shape),
            "raw_action": as_list(raw_action),
            "command": {k: as_list(v) for k, v in command.items()},
            "reward": as_list(reward),
            "terminated": np.asarray(terminated).astype(bool).tolist(),
            "truncated": np.asarray(truncated).astype(bool).tolist(),
            "success_info": info,
            "episode_step_count": int(task.episode_step_count),
            "gripper_qpos_after": {
                "left_gripper": as_list(rv.get_move_group("left_gripper").joint_pos),
                "right_gripper": as_list(rv.get_move_group("right_gripper").joint_pos),
            },
            "raw_action_log_path": cfg.policy_config.raw_action_log_path,
            "debug_dump_dir": cfg.policy_config.debug_dump_dir,
        }
        out = Path(
            "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_act_smoke_report.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
