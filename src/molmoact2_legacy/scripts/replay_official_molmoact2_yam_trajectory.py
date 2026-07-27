from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ARM_MAX_ERROR_RAD = 0.15
DEFAULT_TCP_DISPLACEMENT_ERROR_M = 0.08
DEFAULT_OBJECT_DISPLACEMENT_ERROR_M = 0.05
OBJECT_NAME_MAP = {
    "073-a_lego_duplo": "obj_073-a_lego_duplo",
    "056_tennis_ball": "obj_056_tennis_ball",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an exported official MolmoAct2 YAM trajectory in MolmoSpaces."
    )
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/molmospaces/molmoact2_official_conformance"),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--arm-max-error-rad", type=float, default=DEFAULT_ARM_MAX_ERROR_RAD)
    parser.add_argument(
        "--tcp-displacement-error-m",
        type=float,
        default=DEFAULT_TCP_DISPLACEMENT_ERROR_M,
    )
    parser.add_argument(
        "--object-displacement-error-m",
        type=float,
        default=DEFAULT_OBJECT_DISPLACEMENT_ERROR_M,
    )
    parser.add_argument("--no-os-exit", action="store_true")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    for name in (
        "arm_max_error_rad",
        "tcp_displacement_error_m",
        "object_displacement_error_m",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def scalar_bool(value: Any) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar boolean value, got shape {array.shape}")
    return bool(array.reshape(-1)[0])


def canonical_object_name(name: str) -> str:
    return OBJECT_NAME_MAP.get(name, name)


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            action = np.asarray(row.get("raw_model_action14"), dtype=np.float32)
            if action.shape != (14,):
                raise ValueError(
                    f"{path}:{line_number}: expected raw_model_action14 shape (14,), "
                    f"got {action.shape}"
                )
            row["raw_model_action14"] = action.tolist()
            rows.append(row)
    if not rows:
        raise ValueError(f"Trajectory is empty: {path}")
    return rows


def pose_position(pose7: Any) -> np.ndarray | None:
    if pose7 is None:
        return None
    value = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if value.size < 3 or not np.isfinite(value[:3]).all():
        return None
    return value[:3]


def displacement_error(
    reference_pose: Any,
    reference_initial_pose: Any,
    actual_pose: Any,
    actual_initial_pose: Any,
) -> float | None:
    positions = [
        pose_position(reference_pose),
        pose_position(reference_initial_pose),
        pose_position(actual_pose),
        pose_position(actual_initial_pose),
    ]
    if any(position is None for position in positions):
        return None
    reference_delta = positions[0] - positions[1]
    actual_delta = positions[2] - positions[3]
    return float(np.linalg.norm(reference_delta - actual_delta))


def first_divergence_reason(
    *,
    arm_max_error_rad: float,
    tcp_errors_m: dict[str, float | None],
    object_errors_m: dict[str, float | None],
    thresholds: dict[str, float],
) -> str | None:
    if arm_max_error_rad > thresholds["arm_max_error_rad"]:
        return "robot_joint_tracking"
    if any(
        error is not None and error > thresholds["tcp_displacement_error_m"]
        for error in tcp_errors_m.values()
    ):
        return "tcp_motion"
    if any(
        error is not None and error > thresholds["object_displacement_error_m"]
        for error in object_errors_m.values()
    ):
        return "object_motion"
    return None


def _pose7_from_summary(summary: dict[str, Any]) -> list[float]:
    return [*summary["position"], *summary["quaternion"]]


def _current_model_state(task: Any, config: Any) -> np.ndarray:
    from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
        build_molmoact2_yam_state,
    )

    robot_view = task._env.current_robot.robot_view
    qpos = {
        group: np.asarray(robot_view.get_move_group(group).joint_pos, dtype=np.float32)
        for group in ("left_arm", "left_gripper", "right_arm", "right_gripper")
    }
    return build_molmoact2_yam_state(
        qpos,
        gripper_max=config.policy_config.gripper_max,
        gripper_open_command=config.policy_config.gripper_open_command,
        gripper_closed_command=config.policy_config.gripper_closed_command,
    )


def _body_pose7(task: Any, body_name: str) -> list[float]:
    data = task._env.current_data
    body_id = data.model.body(body_name).id
    return [*data.xpos[body_id].tolist(), *data.xquat[body_id].tolist()]


def _capture_state(task: Any, config: Any) -> dict[str, Any]:
    info = task.get_info()[0]
    behavior = info["official_yam_behavior"]
    return {
        "model_state": _current_model_state(task, config).tolist(),
        "left_tcp_pose7": _body_pose7(task, "robot_0/left_link_6"),
        "right_tcp_pose7": _body_pose7(task, "robot_0/right_link_6"),
        "tcp_frame": "link_6",
        "objects": {
            name: _pose7_from_summary(pose) for name, pose in behavior["object_poses"].items()
        },
        "task": {
            "success": bool(info["success"]),
            "n_in_box": int(info["n_in_box"]),
            "n_total": int(info["n_total"]),
        },
        "contacts": behavior["contacts"],
        "gripper_object_distances": behavior["gripper_object_distances"],
    }


def _apply_initial_state(task: Any, config: Any, official: dict[str, Any]) -> dict[str, Any]:
    import mujoco

    from molmo_spaces.env.data_views import create_mlspaces_body
    from molmo_spaces.utils.pose import pos_quat_to_pose_mat

    robot_view = task._env.current_robot.robot_view
    model_state = np.asarray(official["model_state"], dtype=np.float64).reshape(14)
    robot_view.get_move_group("left_arm").joint_pos = model_state[:6]
    robot_view.get_move_group("right_arm").joint_pos = model_state[7:13]

    closed = float(config.policy_config.gripper_closed_command)
    opened = float(config.policy_config.gripper_open_command)
    left_gripper = closed + float(model_state[6]) * (opened - closed)
    right_gripper = closed + float(model_state[13]) * (opened - closed)
    robot_view.get_move_group("left_gripper").joint_pos = np.array(
        [left_gripper, left_gripper], dtype=np.float64
    )
    robot_view.get_move_group("right_gripper").joint_pos = np.array(
        [right_gripper, right_gripper], dtype=np.float64
    )

    mapped_objects: dict[str, dict[str, Any]] = {}
    for official_name, official_pose_value in official.get("objects", {}).items():
        local_name = canonical_object_name(official_name)
        official_pose = np.asarray(official_pose_value, dtype=np.float64).reshape(7)
        body = create_mlspaces_body(task._env.current_data, local_name)
        local_z = float(body.position[2])
        body.pose = pos_quat_to_pose_mat(
            [float(official_pose[0]), float(official_pose[1]), local_z],
            official_pose[3:7],
        )
        mapped_objects[official_name] = {
            "local_name": local_name,
            "official_pose7": official_pose.tolist(),
            "applied_pose7": [
                float(official_pose[0]),
                float(official_pose[1]),
                local_z,
                *official_pose[3:7].tolist(),
            ],
            "z_mapping": "preserve MolmoSpaces settled tabletop z",
        }

    mujoco.mj_forward(task._env.current_model, task._env.current_data)
    return {
        "model_state14": model_state.tolist(),
        "objects": mapped_objects,
        "cross_engine_mapping": (
            "joint semantics and object XY/quaternion copied; MuJoCo object Z preserved "
            "because the two simulators use different tabletop world-Z baselines"
        ),
    }


def _lookup_pose(state: dict[str, Any], key: str) -> Any:
    if key in state:
        return state[key]
    return state.get("objects", {}).get(key)


def run_replay(args: argparse.Namespace) -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/molmospaces-matplotlib")
    os.environ.setdefault("NLTK_DATA", "/tmp/molmospaces-nltk-data")
    os.environ.setdefault("MOLMOSPACES_ALLOW_WORDNET_FALLBACK", "1")
    with suppress(Exception):
        import nltk

        nltk.download = lambda *unused_args, **unused_kwargs: True

    from molmo_spaces.data_generation.config.molmoact2_official_sim_eval_yam_box_config import (
        MolmoAct2OfficialSimEvalYamBoxDataGenConfig,
    )
    from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
        execute_molmoact2_yam_action,
        molmoact2_yam_action_to_move_group_command,
    )

    trajectory_path = args.trajectory.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else trajectory_path.with_name("manifest.json")
    )
    rows = load_trajectory(trajectory_path)
    if args.max_steps is not None:
        rows = rows[: args.max_steps]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.artifact_root / f"open_loop_replay_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    steps_path = output_dir / "steps.jsonl"
    report_path = output_dir / "report.json"

    thresholds = {
        "arm_max_error_rad": float(args.arm_max_error_rad),
        "tcp_displacement_error_m": float(args.tcp_displacement_error_m),
        "object_displacement_error_m": float(args.object_displacement_error_m),
    }
    report: dict[str, Any] = {
        "status": "started",
        "evidence_level": "cross-simulator open-loop behavior conformance diagnostic",
        "tcp_comparison_frame": "link_6 (official YAM tcp link; not grasp_site)",
        "trajectory_path": str(trajectory_path),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "steps_path": str(steps_path),
        "thresholds": thresholds,
        "requested_steps": len(rows),
        "executed_steps": 0,
        "first_divergence": None,
        "official_success_observed": False,
        "molmospaces_success_observed": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    sampler = None
    try:
        config = MolmoAct2OfficialSimEvalYamBoxDataGenConfig()
        config.task_horizon = len(rows) + 2
        sampler = config.task_sampler_config.task_sampler_class(config)
        task = sampler.sample_task(force_advance_scene=True, house_index=0)

        mapping = _apply_initial_state(task, config, manifest["initial_state"])
        official_initial = manifest["initial_state"]
        actual_initial = _capture_state(task, config)
        report["initial_state_mapping"] = mapping
        report["molmospaces_initial_state"] = actual_initial

        with steps_path.open("w", encoding="utf-8") as output:
            for replay_index, official_row in enumerate(rows):
                raw_action = np.asarray(official_row["raw_model_action14"], dtype=np.float32)
                command = molmoact2_yam_action_to_move_group_command(
                    raw_action,
                    gripper_max=config.policy_config.gripper_max,
                    gripper_open_command=config.policy_config.gripper_open_command,
                    gripper_closed_command=config.policy_config.gripper_closed_command,
                    grasping_type=config.policy_config.grasping_type,
                    grasping_threshold=config.policy_config.grasping_threshold,
                )
                _, reward, terminated, truncated, _ = execute_molmoact2_yam_action(
                    task,
                    command,
                    execution_mode=config.policy_config.execution_mode,
                    command_hz=config.policy_config.execution_command_hz,
                )
                actual = _capture_state(task, config)
                reference = official_row["post"]
                target_state = np.asarray(reference["model_state"], dtype=np.float64)
                actual_state = np.asarray(actual["model_state"], dtype=np.float64)
                arm_errors = np.abs(
                    np.concatenate(
                        [
                            target_state[:6] - actual_state[:6],
                            target_state[7:13] - actual_state[7:13],
                        ]
                    )
                )

                tcp_errors = {
                    side: displacement_error(
                        reference.get(f"{side}_tcp_pose7"),
                        official_initial.get(f"{side}_tcp_pose7"),
                        actual.get(f"{side}_tcp_pose7"),
                        actual_initial.get(f"{side}_tcp_pose7"),
                    )
                    for side in ("left", "right")
                }
                object_errors: dict[str, float | None] = {}
                for official_name, initial_pose in official_initial.get("objects", {}).items():
                    local_name = canonical_object_name(official_name)
                    reference_pose = reference.get("objects", {}).get(official_name)
                    actual_pose = actual.get("objects", {}).get(local_name)
                    actual_initial_pose = actual_initial.get("objects", {}).get(local_name)
                    object_errors[official_name] = displacement_error(
                        reference_pose,
                        initial_pose,
                        actual_pose,
                        actual_initial_pose,
                    )

                reason = first_divergence_reason(
                    arm_max_error_rad=float(arm_errors.max(initial=0.0)),
                    tcp_errors_m=tcp_errors,
                    object_errors_m=object_errors,
                    thresholds=thresholds,
                )
                official_task = reference.get("task", {})
                official_success = scalar_bool(official_task.get("success", False))
                actual_success = bool(actual["task"]["success"])
                report["official_success_observed"] |= official_success
                report["molmospaces_success_observed"] |= actual_success
                if report["first_divergence"] is None and reason is not None:
                    report["first_divergence"] = {
                        "replay_index": replay_index,
                        "official_step": official_row.get("step"),
                        "chunk_id": official_row.get("chunk_id"),
                        "chunk_index": official_row.get("chunk_index"),
                        "reason": reason,
                    }

                replay_row = {
                    "replay_index": replay_index,
                    "official_step": official_row.get("step"),
                    "chunk_id": official_row.get("chunk_id"),
                    "chunk_index": official_row.get("chunk_index"),
                    "raw_model_action14": raw_action.tolist(),
                    "reference_model_state14": target_state.tolist(),
                    "actual_model_state14": actual_state.tolist(),
                    "arm_tracking": {
                        "max_abs_error_rad": float(arm_errors.max(initial=0.0)),
                        "l2_error_rad": float(np.linalg.norm(arm_errors)),
                    },
                    "tcp_displacement_errors_m": tcp_errors,
                    "object_displacement_errors_m": object_errors,
                    "divergence_reason": reason,
                    "official_task": official_task,
                    "molmospaces": actual,
                    "reward": to_jsonable(reward),
                    "terminated": to_jsonable(terminated),
                    "truncated": to_jsonable(truncated),
                }
                output.write(json.dumps(replay_row, separators=(",", ":")) + "\n")
                output.flush()
                report["executed_steps"] = replay_index + 1
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                    break

        report["status"] = "completed"
        report["behavior_equivalent_within_thresholds"] = report["first_divergence"] is None
        report["finished_at_epoch"] = time.time()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        report["finished_at_epoch"] = time.time()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1
    finally:
        if sampler is not None and args.no_os_exit:
            with suppress(Exception):
                sampler.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code = run_replay(args)
    if args.no_os_exit:
        return code
    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
