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

ARTIFACT_ROOT = Path("artifacts/molmospaces/molmoact2_yam_integration")
DEFAULT_ENDPOINT = "http://127.0.0.1:8202/act"
DEFAULT_MAX_STEPS = 30
DEFAULT_ACTION_SCALE = 4.0
DEFAULT_RIGHT_J3_OFFSET = -0.08
MAX_EXECUTED_STEPS = 30
LEFT_ARM_SLICE = slice(0, 6)
RIGHT_ARM_SLICE = slice(7, 13)
LEFT_GRIPPER_ACTION_INDEX = 6
RIGHT_GRIPPER_ACTION_INDEX = 13
# Matches the existing model-in-loop probe's right_j3 convention:
# joint index 3 within the right arm, which maps to 14D action index 10.
RIGHT_J3_ACTION_INDEX = 10
MAX_ABS_ARM_DELTA = 0.24
DIAGNOSTIC_EVIDENCE_LABEL = (
    "diagnostic evidence only: temporary corrected-adapter candidates; "
    "not an official reproduction or adapter semantic change"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded MolmoAct2 official sim_eval YAM diagnostic against an "
            "already-running /act service."
        )
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--max-steps", type=_step_limit, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--action-scale", type=float, default=DEFAULT_ACTION_SCALE)
    parser.add_argument("--right-j3-offset", type=float, default=DEFAULT_RIGHT_J3_OFFSET)
    parser.add_argument("--artifact-root", default=str(ARTIFACT_ROOT))
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--debug-dump-max-calls", type=int, default=1)
    parser.add_argument(
        "--gripper-continuity-guard-steps",
        type=int,
        default=0,
        help="Clamp opening beyond the last executed command for the first N chunk steps while transporting/holding.",
    )
    parser.add_argument("--structured-step-diagnostics", action="store_true")
    parser.add_argument("--include-mujoco-forces", action="store_true")
    parser.add_argument("--no-os-exit", action="store_true")
    args = parser.parse_args(argv)
    if args.gripper_continuity_guard_steps < 0:
        parser.error("--gripper-continuity-guard-steps must be non-negative")
    return args


def _step_limit(value: str) -> int:
    steps = int(value)
    if steps < 1 or steps > MAX_EXECUTED_STEPS:
        raise argparse.ArgumentTypeError(
            f"max steps must be between 1 and {MAX_EXECUTED_STEPS}, got {steps}"
        )
    return steps


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def make_artifact_dir(root: str | Path, *, timestamp: str | None = None) -> Path:
    stamp = timestamp or globals()["timestamp"]()
    return Path(root) / f"corrected_adapter_diagnostic_{stamp}"


def flat(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).astype(float).tolist()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def apply_corrected_adapter_candidates(
    raw_action: Any,
    base_state14: Any,
    *,
    action_scale: float,
    right_j3_offset: float,
    max_abs_arm_delta: float = MAX_ABS_ARM_DELTA,
) -> np.ndarray:
    """Apply temporary diagnostic candidates without changing adapter semantics."""

    action = np.asarray(raw_action, dtype=np.float32).copy().reshape(14)
    base = np.asarray(base_state14, dtype=np.float32).reshape(14)

    if action_scale != 1.0:
        for arm_slice in (LEFT_ARM_SLICE, RIGHT_ARM_SLICE):
            delta = np.clip(
                action[arm_slice] - base[arm_slice], -max_abs_arm_delta, max_abs_arm_delta
            )
            action[arm_slice] = base[arm_slice] + np.clip(
                delta * float(action_scale),
                -max_abs_arm_delta,
                max_abs_arm_delta,
            )

    action[RIGHT_J3_ACTION_INDEX] = base[RIGHT_J3_ACTION_INDEX] + float(right_j3_offset)
    return action.astype(np.float32)


def apply_gripper_continuity_guard(
    action: Any,
    *,
    chunk_index: int,
    guard_steps: int,
    transport_active: bool,
    last_executed_grippers: dict[str, float] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    guarded = np.asarray(action, dtype=np.float32).copy().reshape(14)
    enabled = int(guard_steps) > 0
    within_window = 0 <= int(chunk_index) < int(guard_steps)
    metadata: dict[str, Any] = {
        "enabled": enabled,
        "guard_steps": int(guard_steps),
        "within_window": within_window,
        "transport_active": bool(transport_active),
        "applied": False,
        "clamped_sides": [],
    }
    if not (enabled and within_window and transport_active and last_executed_grippers):
        return guarded, metadata

    for side, index in (
        ("left", LEFT_GRIPPER_ACTION_INDEX),
        ("right", RIGHT_GRIPPER_ACTION_INDEX),
    ):
        previous = finite_float(last_executed_grippers.get(side))
        if previous is not None and float(guarded[index]) > previous:
            guarded[index] = previous
            metadata["clamped_sides"].append(side)
    metadata["applied"] = bool(metadata["clamped_sides"])
    return guarded, metadata


def _pose_position(poses: Any, preferred_name: str | None = None) -> np.ndarray | None:
    if not isinstance(poses, dict) or not poses:
        return None
    pose = poses.get(preferred_name) if preferred_name else None
    if not isinstance(pose, dict):
        pose = next((value for value in poses.values() if isinstance(value, dict)), None)
    if not isinstance(pose, dict) or "position" not in pose:
        return None
    position = np.asarray(pose["position"], dtype=np.float64).reshape(-1)
    return position[:3] if position.size >= 3 and np.isfinite(position[:3]).all() else None


def _relative_motion(
    previous_summary: dict[str, Any] | None,
    summary: dict[str, Any],
) -> dict[str, float | None]:
    if not previous_summary:
        return {
            "object_displacement_m": None,
            "tcp_displacement_m": None,
            "relative_slip_m": None,
        }
    previous_object = _pose_position(previous_summary.get("object_poses"))
    current_object = _pose_position(summary.get("object_poses"))
    previous_tcp = _pose_position(previous_summary.get("gripper_poses"), "right_gripper")
    current_tcp = _pose_position(summary.get("gripper_poses"), "right_gripper")
    object_delta = (
        None
        if previous_object is None or current_object is None
        else current_object - previous_object
    )
    tcp_delta = None if previous_tcp is None or current_tcp is None else current_tcp - previous_tcp
    return {
        "object_displacement_m": None
        if object_delta is None
        else float(np.linalg.norm(object_delta)),
        "tcp_displacement_m": None if tcp_delta is None else float(np.linalg.norm(tcp_delta)),
        "relative_slip_m": (
            None
            if object_delta is None or tcp_delta is None
            else float(np.linalg.norm(object_delta - tcp_delta))
        ),
    }


def build_structured_step_diagnostic(
    *,
    step: int,
    chunk_id: int,
    chunk_index: int,
    raw_action: Any,
    mapped_action: Any,
    actual_state14: Any,
    summary: dict[str, Any],
    previous_summary: dict[str, Any] | None,
    guard: dict[str, Any],
    force_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = np.asarray(raw_action, dtype=np.float64).reshape(14)
    mapped = np.asarray(mapped_action, dtype=np.float64).reshape(14)
    actual = np.asarray(actual_state14, dtype=np.float64).reshape(14)
    forces = force_diagnostics or {"available": False, "reason": "not_requested"}
    return {
        "schema_version": 1,
        "step": int(step),
        "chunk": {"id": int(chunk_id), "index": int(chunk_index)},
        "gripper": {
            "raw": {"left": float(raw[6]), "right": float(raw[13])},
            "mapped": {"left": float(mapped[6]), "right": float(mapped[13])},
            "actual_normalized_opening": {
                "left": float(actual[6]),
                "right": float(actual[13]),
            },
            "guard": guard,
        },
        "actual_normalized_state": flat(actual),
        "transport": {
            "touching": bool(summary.get("any_gripper_contact", False)),
            "held": bool(summary.get("held", False)),
        },
        "poses": {
            "objects": summary.get("object_poses", {}),
            "tcp": summary.get("gripper_poses", {}),
        },
        "relative_motion": _relative_motion(previous_summary, summary),
        "target_minus_post_step_normalized_state": {
            "left": flat(mapped[LEFT_ARM_SLICE] - actual[LEFT_ARM_SLICE]),
            "right": flat(mapped[RIGHT_ARM_SLICE] - actual[RIGHT_ARM_SLICE]),
            "left_norm": float(np.linalg.norm(mapped[LEFT_ARM_SLICE] - actual[LEFT_ARM_SLICE])),
            "right_norm": float(np.linalg.norm(mapped[RIGHT_ARM_SLICE] - actual[RIGHT_ARM_SLICE])),
        },
        "contacts": summary.get("contacts", {}),
        "forces": forces,
    }


def collect_optional_mujoco_forces(task: Any, *, requested: bool) -> dict[str, Any]:
    if not requested:
        return {"available": False, "reason": "not_requested", "source": None}
    try:
        data = None
        source = None
        for env_name in ("_env", "env"):
            env = getattr(task, env_name, None)
            if env is None:
                continue
            current_data = getattr(env, "current_data", None)
            if current_data is not None:
                data = current_data
                source = f"task.{env_name}.current_data"
                break
            mj_datas = getattr(env, "mj_datas", None)
            if mj_datas is not None:
                batch_index = int(getattr(task, "current_batch_index", 0) or 0)
                try:
                    data = mj_datas[batch_index]
                    source = f"task.{env_name}.mj_datas[{batch_index}]"
                    break
                except (IndexError, KeyError, TypeError):
                    pass
        if data is None:
            controller = getattr(task, "controller", None)
            physics = getattr(controller, "physics", None) or getattr(task, "physics", None)
            data = getattr(physics, "data", None)
            if data is not None:
                source = (
                    "task.controller.physics.data"
                    if controller is not None
                    else "task.physics.data"
                )
        if data is None:
            return {
                "available": False,
                "reason": "mujoco_data_unavailable",
                "source": None,
            }
        actuator_force = getattr(data, "actuator_force", None)
        efc_force = getattr(data, "efc_force", None)
        if actuator_force is None and efc_force is None:
            return {
                "available": False,
                "reason": "force_arrays_unavailable",
                "source": source,
            }
        return {
            "available": True,
            "source": source,
            "actuator_force": {
                "available": actuator_force is not None,
                "scope": "global_actuator_array",
                "values": None if actuator_force is None else flat(actuator_force),
            },
            "efc_force": {
                "available": efc_force is not None,
                "scope": "global_unscoped_constraint_force_array",
                "values": None if efc_force is None else flat(efc_force),
            },
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": "force_collection_failed",
            "source": None,
            "error": repr(exc),
        }


def summarize_info(info: Any) -> dict[str, Any]:
    item = info[0] if isinstance(info, list) else info
    item = item if isinstance(item, dict) else {}
    behavior = item.get("official_yam_behavior", {}) or {}
    distances = behavior.get("gripper_object_distances", {}) or {}
    contacts = behavior.get("contacts", {}) or {}

    all_distances: list[float] = []
    for per_gripper in distances.values():
        if isinstance(per_gripper, dict):
            for value in per_gripper.values():
                number = finite_float(value)
                if number is not None:
                    all_distances.append(number)

    any_gripper_contact = False
    held = False
    for obj_contact in contacts.values():
        if isinstance(obj_contact, dict):
            any_gripper_contact = bool(
                any_gripper_contact or obj_contact.get("any_gripper_touching")
            )
            held = bool(
                held
                or obj_contact.get("held")
                or obj_contact.get("is_held")
                or obj_contact.get("grasped")
            )
            per_gripper = obj_contact.get("per_gripper", {}) or {}
            if isinstance(per_gripper, dict):
                any_gripper_contact = bool(
                    any_gripper_contact
                    or any(
                        isinstance(gripper_contact, dict) and gripper_contact.get("touching", False)
                        for gripper_contact in per_gripper.values()
                    )
                )
                held = bool(
                    held
                    or any(
                        isinstance(gripper_contact, dict) and gripper_contact.get("held", False)
                        for gripper_contact in per_gripper.values()
                    )
                )

    n_in_box = item.get("n_in_box")
    if n_in_box is None:
        official_box = item.get("official_yam_box", {}) or {}
        n_in_box = official_box.get("n_in_box")

    return {
        "success": bool(item.get("success", False)),
        "n_in_box": None if n_in_box is None else int(n_in_box),
        "min_distance_m": min(all_distances) if all_distances else None,
        "gripper_object_distances": distances,
        "contacts": contacts,
        "any_gripper_contact": bool(any_gripper_contact),
        "held": bool(held or behavior.get("held", False)),
        "object_poses": behavior.get("object_poses", {}),
        "gripper_poses": behavior.get("gripper_poses", {}),
    }


def write_artifacts(
    out_dir: Path,
    report: dict[str, Any],
    selected_action_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    selected_actions_path = out_dir / "selected_actions.jsonl"

    report = dict(report)
    report["report_path"] = str(report_path)
    report["selected_actions_path"] = str(selected_actions_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    with selected_actions_path.open("w", encoding="utf-8") as handle:
        for row in selected_action_rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return {"report": report_path, "selected_actions": selected_actions_path}


def _norm_delta(action: np.ndarray, base_state14: np.ndarray, arm_slice: slice) -> float:
    return float(np.linalg.norm(action[arm_slice] - base_state14[arm_slice]))


def _make_initial_report(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    return {
        "status": "started",
        "evidence_label": DIAGNOSTIC_EVIDENCE_LABEL,
        "artifact_dir": str(out_dir),
        "endpoint": args.endpoint,
        "constraints": {
            "max_executed_steps": int(args.max_steps),
            "no_long_rollout": True,
            "act_service_restart_requested": False,
            "act_service_stop_requested": False,
            "source_adapter_semantics_modified": False,
        },
        "candidate_parameters": {
            "action_scale": float(args.action_scale),
            "right_j3_offset": float(args.right_j3_offset),
            "right_j3_action_index": RIGHT_J3_ACTION_INDEX,
            "right_j3_convention": "zero-based joint index 3 within right arm, action index 10",
            "max_abs_arm_delta": MAX_ABS_ARM_DELTA,
            "gripper_continuity_guard_steps": int(args.gripper_continuity_guard_steps),
            "structured_step_diagnostics": bool(args.structured_step_diagnostics),
            "include_mujoco_forces": bool(args.include_mujoco_forces),
        },
        "act_response_present": False,
        "selected_action_shape": None,
        "selected_action_shape_before_limit": None,
        "executed_steps": 0,
        "eef_object_min_distance_m": None,
        "eef_object_min_distances_m": [],
        "contact_flags": {"any_gripper_contact": False},
        "final_n_in_box": None,
        "success": None,
        "raw_action_log_path": str(out_dir / "raw_act_response.jsonl"),
        "debug_dump_dir": str(out_dir / "debug_request"),
        "teardown_note": (
            "The diagnostic attaches to the existing /act service and does not start, "
            "stop, or restart it. os._exit is used by default after final flush because "
            "sampler.close/env cleanup has historically hung on this host."
        ),
    }


def run_diagnostic(args: argparse.Namespace) -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/molmospaces-matplotlib")
    os.environ.setdefault("NLTK_DATA", "/tmp/molmospaces-nltk-data")
    os.environ.setdefault("MOLMOSPACES_ALLOW_WORDNET_FALLBACK", "1")
    try:
        import nltk

        nltk.download = lambda *a, **k: True
    except Exception:
        pass

    from molmo_spaces.data_generation.config.molmoact2_official_sim_eval_yam_box_config import (
        MolmoAct2OfficialSimEvalYamBoxDataGenConfig,
    )
    from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
        MolmoAct2YamPolicy,
        execute_molmoact2_yam_action,
    )

    out_dir = make_artifact_dir(args.artifact_root, timestamp=args.timestamp)
    selected_action_rows: list[dict[str, Any]] = []
    report = _make_initial_report(args, out_dir)
    write_artifacts(out_dir, report, selected_action_rows)

    sampler = None
    try:
        cfg = MolmoAct2OfficialSimEvalYamBoxDataGenConfig()
        cfg.task_horizon = max(int(args.max_steps) + 2, 8)
        cfg.policy_config.remote_config = {"host": "127.0.0.1", "port": 8202, "path": "/act"}
        cfg.policy_config.endpoint_url = args.endpoint
        cfg.policy_config.timeout = float(args.timeout)
        cfg.policy_config.num_steps = int(args.max_steps)
        cfg.policy_config.n_action_steps = None
        cfg.policy_config.execution_mode = "sim_eval_step"
        cfg.policy_config.execution_command_hz = None
        cfg.policy_config.raw_action_log_path = str(out_dir / "raw_act_response.jsonl")
        cfg.policy_config.debug_dump_dir = str(out_dir / "debug_request")
        cfg.policy_config.debug_dump_max_calls = int(args.debug_dump_max_calls)

        sampler = cfg.task_sampler_config.task_sampler_class(cfg)
        task = sampler.sample_task(force_advance_scene=True, house_index=0)
        policy = MolmoAct2YamPolicy(cfg, task=task)

        obs = task.get_observations()
        request = policy.obs_to_model_input(obs)
        base_state14 = np.asarray(request["state"], dtype=np.float32).reshape(14)
        raw_action = policy.inference_model(request)
        actions_buffer = policy.actions_buffer or [np.asarray(raw_action, dtype=np.float32)]
        selected_actions = np.asarray(actions_buffer, dtype=np.float32)
        report["act_response_present"] = True
        report["endpoint"] = policy.endpoint_url
        report["request_num_steps"] = request.get("num_steps")
        report["selected_action_shape_before_limit"] = list(selected_actions.shape)
        selected_actions = selected_actions[: int(args.max_steps)]
        report["selected_action_shape"] = list(selected_actions.shape)
        write_artifacts(out_dir, report, selected_action_rows)

        last_executed_grippers: dict[str, float] | None = None
        previous_summary: dict[str, Any] | None = None
        chunk_id = 0
        for step_idx, raw in enumerate(selected_actions, start=1):
            transformed = apply_corrected_adapter_candidates(
                raw,
                base_state14,
                action_scale=float(args.action_scale),
                right_j3_offset=float(args.right_j3_offset),
            )
            transformed, guard = apply_gripper_continuity_guard(
                transformed,
                chunk_index=step_idx - 1,
                guard_steps=int(args.gripper_continuity_guard_steps),
                transport_active=bool(
                    previous_summary
                    and (
                        previous_summary.get("any_gripper_contact", False)
                        or previous_summary.get("held", False)
                    )
                ),
                last_executed_grippers=last_executed_grippers,
            )
            command = policy.model_output_to_action(transformed)
            _observation, reward, terminated, truncated, info = execute_molmoact2_yam_action(
                task,
                command,
                execution_mode=cfg.policy_config.execution_mode,
                joint_step=0.02,
                max_smoothing_steps=3,
                command_hz=cfg.policy_config.execution_command_hz,
            )
            summary = summarize_info(info)
            actual_state14 = np.asarray(
                policy.obs_to_model_input(task.get_observations())["state"],
                dtype=np.float32,
            ).reshape(14)
            if summary["min_distance_m"] is not None:
                old_min = report["eef_object_min_distance_m"]
                report["eef_object_min_distance_m"] = (
                    summary["min_distance_m"]
                    if old_min is None
                    else min(float(old_min), float(summary["min_distance_m"]))
                )
            report["eef_object_min_distances_m"].append(summary["min_distance_m"])
            report["executed_steps"] = step_idx
            report["contact_flags"] = {
                "any_gripper_contact": bool(
                    report["contact_flags"]["any_gripper_contact"] or summary["any_gripper_contact"]
                ),
                "latest_any_gripper_contact": bool(summary["any_gripper_contact"]),
            }
            report["final_n_in_box"] = summary["n_in_box"]
            report["success"] = bool(summary["success"])
            report["status"] = "running"

            row = {
                "step": step_idx,
                "raw_action": flat(raw),
                "transformed_action": flat(transformed),
                "raw_left_delta_norm": _norm_delta(raw, base_state14, LEFT_ARM_SLICE),
                "raw_right_delta_norm": _norm_delta(raw, base_state14, RIGHT_ARM_SLICE),
                "transformed_left_delta_norm": _norm_delta(
                    transformed,
                    base_state14,
                    LEFT_ARM_SLICE,
                ),
                "transformed_right_delta_norm": _norm_delta(
                    transformed,
                    base_state14,
                    RIGHT_ARM_SLICE,
                ),
                "right_j3_raw": float(raw[RIGHT_J3_ACTION_INDEX]),
                "right_j3_transformed": float(transformed[RIGHT_J3_ACTION_INDEX]),
                "reward": flat(reward),
                "terminated": np.asarray(terminated).astype(bool).reshape(-1).tolist(),
                "truncated": np.asarray(truncated).astype(bool).reshape(-1).tolist(),
                "min_distance_m": summary["min_distance_m"],
                "any_gripper_contact": summary["any_gripper_contact"],
                "n_in_box": summary["n_in_box"],
                "success": summary["success"],
                "distances": summary["gripper_object_distances"],
                "contacts": summary["contacts"],
                "gripper_poses": summary["gripper_poses"],
                "object_poses": summary["object_poses"],
            }
            if args.structured_step_diagnostics:
                row["structured_step_diagnostic"] = build_structured_step_diagnostic(
                    step=step_idx,
                    chunk_id=chunk_id,
                    chunk_index=step_idx - 1,
                    raw_action=raw,
                    mapped_action=transformed,
                    actual_state14=actual_state14,
                    summary=summary,
                    previous_summary=previous_summary,
                    guard=guard,
                    force_diagnostics=collect_optional_mujoco_forces(
                        task,
                        requested=bool(args.include_mujoco_forces),
                    ),
                )
            selected_action_rows.append(row)
            write_artifacts(out_dir, report, selected_action_rows)
            last_executed_grippers = {
                "left": float(transformed[LEFT_GRIPPER_ACTION_INDEX]),
                "right": float(transformed[RIGHT_GRIPPER_ACTION_INDEX]),
            }
            previous_summary = summary

            if (
                summary["success"]
                or bool(np.asarray(terminated).any())
                or bool(np.asarray(truncated).any())
                or step_idx >= int(args.max_steps)
            ):
                break

        report["status"] = "completed"
        report["finished_at_epoch"] = time.time()
        write_artifacts(out_dir, report, selected_action_rows)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "artifact_dir": str(out_dir),
                    "executed_steps": report["executed_steps"],
                    "selected_action_shape": report["selected_action_shape"],
                    "eef_object_min_distance_m": report["eef_object_min_distance_m"],
                    "contact_flags": report["contact_flags"],
                    "final_n_in_box": report["final_n_in_box"],
                    "success": report["success"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        report["status"] = "failed"
        report["finished_at_epoch"] = time.time()
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        write_artifacts(out_dir, report, selected_action_rows)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "artifact_dir": str(out_dir),
                    "error": report["error"],
                    "executed_steps": report["executed_steps"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    finally:
        if sampler is not None and args.no_os_exit:
            with suppress(Exception):
                sampler.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code = run_diagnostic(args)
    if args.no_os_exit:
        return code
    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
