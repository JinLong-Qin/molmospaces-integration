from __future__ import annotations

import json, time
from pathlib import Path
import numpy as np

from molmo_spaces.data_generation.config.molmoact2_official_sim_eval_yam_box_config import (
    MolmoAct2OfficialSimEvalYamBoxDataGenConfig,
)
from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
    MolmoAct2YamPolicy,
    execute_molmoact2_yam_action,
)

MAX_ACTIONS = 30


def as_list(x):
    return np.asarray(x, dtype=float).reshape(-1).tolist()


def summarize_info(info):
    item = info[0] if isinstance(info, list) else info
    behavior = item.get("official_yam_behavior", {})
    distances = behavior.get("gripper_object_distances", {})
    contacts = behavior.get("contacts", {})
    return {
        "success": bool(item.get("success", False)),
        "n_in_box": int(item.get("n_in_box", 0)),
        "distances": distances,
        "contacts": contacts,
        "object_poses": behavior.get("object_poses", {}),
        "gripper_poses": behavior.get("gripper_poses", {}),
    }


def min_distance(summary):
    values = []
    for gripper in summary.get("distances", {}).values():
        for value in gripper.values():
            values.append(float(value))
    return min(values) if values else None


def any_gripper_contact(summary):
    for obj in summary.get("contacts", {}).values():
        if obj.get("any_gripper_touching"):
            return True
    return False


def main():
    cfg = MolmoAct2OfficialSimEvalYamBoxDataGenConfig()
    cfg.policy_config.num_steps = MAX_ACTIONS
    cfg.policy_config.n_action_steps = MAX_ACTIONS
    cfg.policy_config.timeout = 120.0
    cfg.policy_config.raw_action_log_path = "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_act30_raw_actions.jsonl"
    cfg.policy_config.debug_dump_dir = (
        "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_act30_debug"
    )
    cfg.policy_config.debug_dump_max_calls = 1
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    try:
        task = sampler.sample_task(force_advance_scene=True, house_index=0)
        policy = MolmoAct2YamPolicy(cfg, task=task)
        obs = task.get_observations()
        request = policy.obs_to_model_input(obs)
        t0 = time.time()
        first_raw = policy.inference_model(request)
        inference_elapsed = time.time() - t0
        raw_actions = [np.asarray(first_raw, dtype=np.float32)]
        first_command = policy.model_output_to_action(first_raw)
        trace = []
        raw = first_raw
        command = first_command
        for step_idx in range(MAX_ACTIONS):
            if step_idx > 0:
                if policy.actions_buffer is None or policy.current_buffer_index >= len(
                    policy.actions_buffer
                ):
                    break
                raw = policy.inference_model(request)
                raw_actions.append(np.asarray(raw, dtype=np.float32))
                command = policy.model_output_to_action(raw)
            observation, reward, terminated, truncated, info = execute_molmoact2_yam_action(
                task,
                command,
                joint_step=0.02,
                max_smoothing_steps=3,
                command_hz=cfg.policy_config.execution_command_hz,
            )
            summary = summarize_info(info)
            trace.append(
                {
                    "step": step_idx + 1,
                    "raw_action": as_list(raw),
                    "reward": as_list(reward),
                    "terminated": np.asarray(terminated).astype(bool).tolist(),
                    "truncated": np.asarray(truncated).astype(bool).tolist(),
                    "min_gripper_object_distance": min_distance(summary),
                    "any_gripper_contact": any_gripper_contact(summary),
                    "summary": summary,
                }
            )
            if (
                summary["success"]
                or bool(np.asarray(terminated).any())
                or bool(np.asarray(truncated).any())
            ):
                break
        min_dist = min(
            [
                x["min_gripper_object_distance"]
                for x in trace
                if x["min_gripper_object_distance"] is not None
            ],
            default=None,
        )
        report = {
            "evidence_level": "single /act chunk with <=30 selected actions executed for bounded diagnostic; not a long rollout",
            "endpoint_url": policy.endpoint_url,
            "request_num_steps": request.get("num_steps"),
            "inference_elapsed_sec": inference_elapsed,
            "actions_returned": 0 if policy.actions_buffer is None else len(policy.actions_buffer),
            "actions_executed": len(trace),
            "raw_actions_shape": list(np.asarray(raw_actions).shape),
            "any_success": any(x["summary"]["success"] for x in trace),
            "any_gripper_contact": any(x["any_gripper_contact"] for x in trace),
            "min_gripper_object_distance": min_dist,
            "final_n_in_box": trace[-1]["summary"]["n_in_box"] if trace else None,
            "trace": trace,
            "raw_action_log_path": cfg.policy_config.raw_action_log_path,
            "debug_dump_dir": cfg.policy_config.debug_dump_dir,
        }
        out = Path(
            "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_act30_diagnostic_report.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        compact = {
            k: report[k]
            for k in [
                "evidence_level",
                "endpoint_url",
                "request_num_steps",
                "inference_elapsed_sec",
                "actions_returned",
                "actions_executed",
                "any_success",
                "any_gripper_contact",
                "min_gripper_object_distance",
                "final_n_in_box",
                "raw_action_log_path",
                "debug_dump_dir",
            ]
        }
        print(json.dumps(compact, indent=2, sort_keys=True, default=str))
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
