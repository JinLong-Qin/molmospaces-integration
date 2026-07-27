"""Bounded env-construction smoke for the official sim_eval YAM bridge.

No policy call and no rollout: this only compiles the MolmoSpaces scene with the
official MolmoAct2 YAM MJCF, samples one task, renders camera observations if
available, and writes a diagnostic evidence report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from molmo_spaces.data_generation.config.molmoact2_official_sim_eval_yam_box_config import (
    MolmoAct2OfficialSimEvalYamBoxDataGenConfig,
)


def _as_list(value):
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def _collect_image_like(summary: dict, prefix: str, value) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _collect_image_like(summary, f"{prefix}.{key}" if prefix else str(key), child)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _collect_image_like(summary, f"{prefix}[{index}]", child)
        return
    try:
        arr = np.asarray(value)
    except Exception:
        return
    if arr.dtype == object or arr.ndim < 2 or arr.size <= 1000:
        return
    if arr.ndim >= 3 or prefix.endswith("_cam") or "cam" in prefix or "image" in prefix:
        numeric = arr.astype(np.float32, copy=False)
        summary[prefix] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "mean": float(numeric.mean()),
            "std": float(numeric.std()),
        }


def main() -> None:
    config = MolmoAct2OfficialSimEvalYamBoxDataGenConfig()
    sampler = config.task_sampler_config.task_sampler_class(config)
    task = None
    try:
        task = sampler.sample_task(force_advance_scene=True, house_index=0)
        if task is None:
            raise AssertionError("sample_task returned None")
        env = task._env
        data = env.current_data
        model = env.current_model
        robot_view = env.current_robot.robot_view

        def body_pos(name: str):
            return _as_list(data.xpos[model.body(name).id])

        camera_summary = {}
        obs = task.get_observations()[0]
        _collect_image_like(camera_summary, "", obs)

        summary = {
            "evidence_level": "env construction/task sampling/camera observation smoke; no policy /act and no rollout",
            "config": config.__class__.__name__,
            "robot_view": robot_view.__class__.__name__,
            "move_groups": sorted(robot_view.move_group_ids()),
            "base_size": config.robot_config.base_size,
            "body_positions": {
                "robot_0/base": body_pos("robot_0/base")
                if model.body("robot_0/base") is not None
                else None,
                "robot_0/bimanual_base": body_pos("robot_0/bimanual_base"),
                "obj_073-a_lego_duplo": body_pos("obj_073-a_lego_duplo"),
                "obj_056_tennis_ball": body_pos("obj_056_tennis_ball"),
                "open_box": body_pos("open_box"),
            },
            "gripper_qpos": {
                "left_gripper": _as_list(robot_view.get_move_group("left_gripper").joint_pos),
                "right_gripper": _as_list(robot_view.get_move_group("right_gripper").joint_pos),
            },
            "task_description": task.get_task_description(),
            "task_objects": task.get_task_objects(),
            "camera_summary": camera_summary,
        }
        bimanual_z = summary["body_positions"]["robot_0/bimanual_base"][2]
        if abs(bimanual_z - 0.01) > 0.05:
            raise AssertionError(f"bimanual_base z unexpectedly elevated: {bimanual_z}")
        if not camera_summary:
            raise AssertionError("No image-like camera observation found")
        output = Path(
            "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_env_smoke_report.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
