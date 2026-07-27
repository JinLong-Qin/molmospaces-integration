from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/molmospaces-matplotlib")
os.environ.setdefault("NLTK_DATA", "/tmp/molmospaces-nltk-data")
try:
    import nltk

    nltk.download = lambda *args, **kwargs: True
except ImportError:
    pass

import json, os, time, traceback
from pathlib import Path
import numpy as np

from molmo_spaces.data_generation.config.molmoact2_official_sim_eval_yam_box_config import (
    MolmoAct2OfficialSimEvalYamBoxDataGenConfig,
)
from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
    MolmoAct2YamPolicy,
    execute_molmoact2_yam_action,
)

MAX_ACTIONS = 90
CHUNK_SIZE = 5
OUT = Path(
    "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_contact90_stream_report.json"
)


def as_list(x):
    return np.asarray(x, dtype=float).reshape(-1).tolist()


def summarize_info(info):
    item = info[0] if isinstance(info, list) else info
    behavior = item.get("official_yam_behavior", {})
    return {
        "success": bool(item.get("success", False)),
        "n_in_box": int(item.get("n_in_box", 0)),
        "distances": behavior.get("gripper_object_distances", {}),
        "contacts": behavior.get("contacts", {}),
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
    return any(obj.get("any_gripper_touching") for obj in summary.get("contacts", {}).values())


def write_report(report):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["updated_at"] = time.time()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")


def main():
    cfg = MolmoAct2OfficialSimEvalYamBoxDataGenConfig()
    cfg.policy_config.num_steps = CHUNK_SIZE
    cfg.policy_config.n_action_steps = CHUNK_SIZE
    cfg.policy_config.timeout = 120.0
    cfg.policy_config.raw_action_log_path = "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_contact90_stream_raw_actions.jsonl"
    cfg.policy_config.debug_dump_dir = "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_contact90_stream_debug"
    cfg.policy_config.debug_dump_max_calls = 2

    report = {
        "execution_mode": cfg.policy_config.execution_mode,
        "evidence_level": "contact-capped streamed /act diagnostic: chunks of 5 actions, up to 90 selected actions, stops on first gripper-object contact/success, flushes after every action; not a long rollout",
        "endpoint_url": None,
        "chunk_size": CHUNK_SIZE,
        "max_actions": MAX_ACTIONS,
        "chunks_requested": 0,
        "actions_executed": 0,
        "any_success": False,
        "any_gripper_contact": False,
        "min_gripper_object_distance": None,
        "final_n_in_box": None,
        "status": "started",
        "trace": [],
        "raw_action_log_path": cfg.policy_config.raw_action_log_path,
        "debug_dump_dir": cfg.policy_config.debug_dump_dir,
        "teardown_note": "script intentionally uses os._exit after final flush because sampler.close/env cleanup can hang on this host",
    }
    write_report(report)
    try:
        sampler = cfg.task_sampler_config.task_sampler_class(cfg)
        task = sampler.sample_task(force_advance_scene=True, house_index=0)
        policy = MolmoAct2YamPolicy(cfg, task=task)
        report["endpoint_url"] = policy.endpoint_url
        write_report(report)

        while report["actions_executed"] < MAX_ACTIONS:
            obs = task.get_observations()
            request = policy.obs_to_model_input(obs)
            policy.actions_buffer = None
            policy.current_buffer_index = 0
            t0 = time.time()
            first_raw = policy.inference_model(request)
            inference_elapsed = time.time() - t0
            report["chunks_requested"] += 1
            report.setdefault("chunk_inference_elapsed_sec", []).append(inference_elapsed)
            raw = first_raw
            for chunk_idx in range(CHUNK_SIZE):
                if chunk_idx > 0:
                    if policy.actions_buffer is None or policy.current_buffer_index >= len(
                        policy.actions_buffer
                    ):
                        break
                    raw = policy.inference_model(request)
                command = policy.model_output_to_action(raw)
                observation, reward, terminated, truncated, info = execute_molmoact2_yam_action(
                    task,
                    command,
                    execution_mode=cfg.policy_config.execution_mode,
                    joint_step=0.02,
                    max_smoothing_steps=3,
                    command_hz=cfg.policy_config.execution_command_hz,
                )
                summary = summarize_info(info)
                dist = min_distance(summary)
                contact = any_gripper_contact(summary)
                report["actions_executed"] += 1
                report["any_success"] = bool(report["any_success"] or summary["success"])
                report["any_gripper_contact"] = bool(report["any_gripper_contact"] or contact)
                if dist is not None:
                    old = report["min_gripper_object_distance"]
                    report["min_gripper_object_distance"] = (
                        dist if old is None else min(float(old), float(dist))
                    )
                report["final_n_in_box"] = summary["n_in_box"]
                report["trace"].append(
                    {
                        "step": report["actions_executed"],
                        "chunk": report["chunks_requested"],
                        "raw_action": as_list(raw),
                        "reward": as_list(reward),
                        "terminated": np.asarray(terminated).astype(bool).tolist(),
                        "truncated": np.asarray(truncated).astype(bool).tolist(),
                        "min_gripper_object_distance": dist,
                        "any_gripper_contact": contact,
                        "summary": summary,
                    }
                )
                report["status"] = "running"
                write_report(report)
                if (
                    summary["success"]
                    or contact
                    or bool(np.asarray(terminated).any())
                    or bool(np.asarray(truncated).any())
                    or report["actions_executed"] >= MAX_ACTIONS
                ):
                    break
            if (
                report["any_success"]
                or report["any_gripper_contact"]
                or report["actions_executed"] >= MAX_ACTIONS
            ):
                break
        report["status"] = "completed"
        write_report(report)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in [
                        "status",
                        "chunks_requested",
                        "actions_executed",
                        "any_success",
                        "any_gripper_contact",
                        "min_gripper_object_distance",
                        "final_n_in_box",
                    ]
                },
                indent=2,
                sort_keys=True,
            )
        )
        os._exit(0)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        write_report(report)
        print(
            json.dumps(
                {
                    k: report.get(k)
                    for k in ["status", "error", "actions_executed", "min_gripper_object_distance"]
                },
                indent=2,
                sort_keys=True,
            )
        )
        os._exit(1)


if __name__ == "__main__":
    main()
