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

from scripts.diagnose_molmoact2_official_sim_eval_yam_corrected_adapter import (
    ARTIFACT_ROOT,
    DEFAULT_ACTION_SCALE,
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_STEPS,
    DEFAULT_RIGHT_J3_OFFSET,
    DIAGNOSTIC_EVIDENCE_LABEL,
    LEFT_ARM_SLICE,
    MAX_ABS_ARM_DELTA,
    RIGHT_ARM_SLICE,
    RIGHT_J3_ACTION_INDEX,
    apply_corrected_adapter_candidates,
    apply_gripper_continuity_guard,
    build_structured_step_diagnostic,
    collect_optional_mujoco_forces,
    finite_float,
    flat,
    summarize_info,
)
from scripts.diagnose_molmoact2_official_sim_eval_yam_corrected_adapter import (
    MAX_EXECUTED_STEPS as _CORRECTED_MAX_STEPS,
)

MAX_EXECUTED_STEPS = max(200, _CORRECTED_MAX_STEPS)
OFFICIAL_BASELINE_ENDPOINT = "http://127.0.0.1:8203/act"
OFFICIAL_BASELINE_COMMAND_HZ = 30.0

FRAME_PROBE_LABEL = (
    "diagnostic evidence only: bounded action-to-EEF/frame probe for temporary "
    "corrected-adapter candidates; not an official reproduction or adapter semantic change"
)


def _step_limit(value: str) -> int:
    steps = int(value)
    if steps < 1 or steps > MAX_EXECUTED_STEPS:
        raise argparse.ArgumentTypeError(
            f"max steps must be between 1 and {MAX_EXECUTED_STEPS}, got {steps}"
        )
    return steps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded MolmoAct2 official sim_eval YAM action-to-EEF/frame "
            "probe against an already-running /act service."
        )
    )
    parser.add_argument("--endpoint", default=OFFICIAL_BASELINE_ENDPOINT)
    parser.add_argument("--max-steps", type=_step_limit, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--action-scale", type=float, default=DEFAULT_ACTION_SCALE)
    parser.add_argument("--right-j3-offset", type=float, default=DEFAULT_RIGHT_J3_OFFSET)
    parser.add_argument("--artifact-root", default=str(ARTIFACT_ROOT))
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--debug-dump-max-calls", type=int, default=1)
    parser.add_argument(
        "--adapter-option",
        default="passthrough",
        choices=(
            "scale4_rightj3m008",
            "right_j2_delta_negated",
            "bilateral_j2_delta_negated",
            "right_yz_both_negated",
            "right_yz_swapped",
            "right_yz_swapped_both_negated",
            "passthrough",
            "passthrough_gripflip",
        ),
        help=(
            "Named temporary diagnostic adapter option. The default passthrough preserves "
            "the official policy action for the deterministic baseline. "
            "right_j2_delta_negated reproduces the older best bounded axis/sign candidate. "
            "The right_yz_* options are tighter frame probes that only remap/sign-flip the "
            "first three right-arm delta dims before applying the existing right_j3 offset."
        ),
    )
    parser.add_argument("--intended-left-object", default="obj_073-a_lego_duplo")
    parser.add_argument("--intended-right-object", default="obj_056_tennis_ball")
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument(
        "--replan-every",
        type=int,
        default=0,
        help=(
            "If > 0, re-query /act every N executed steps to give the policy a "
            "fresh observation and refresh the action chunk. Default 0 keeps the "
            "old single-chunk open-loop behavior for backward compatibility."
        ),
    )
    parser.add_argument("--no-os-exit", action="store_true")
    parser.add_argument("--gripper-continuity-guard-steps", type=int, default=0)
    parser.add_argument("--structured-step-diagnostics", action="store_true")
    parser.add_argument("--include-mujoco-forces", action="store_true")
    args = parser.parse_args(argv)
    if args.gripper_continuity_guard_steps < 0:
        parser.error("--gripper-continuity-guard-steps must be non-negative")
    return args


def infer_buffered_action_with_chunk_metadata(
    policy: Any,
    model_input: dict[str, Any],
    *,
    chunk_id: int,
) -> tuple[np.ndarray, int, int, bool]:
    """Consume the real policy buffer and return exact chunk boundary metadata."""

    calls_before = int(policy.inference_call_count)
    action = np.asarray(policy.inference_model(model_input), dtype=np.float32).reshape(14)
    calls_after = int(policy.inference_call_count)
    new_chunk = calls_after > calls_before
    if new_chunk:
        chunk_id += calls_after - calls_before
    chunk_index = int(policy.current_buffer_index) - 1
    return action, int(chunk_id), chunk_index, new_chunk


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def make_artifact_dir(root: str | Path, *, stamp: str | None = None) -> Path:
    return Path(root) / f"frame_probe_{stamp or timestamp()}"


def _vector3(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 3:
        return None
    vector = array[:3].astype(float)
    if not np.all(np.isfinite(vector)):
        return None
    return vector


def _pose_position(container: dict[str, Any], name: str) -> np.ndarray | None:
    pose = (container or {}).get(name, {})
    if not isinstance(pose, dict):
        return None
    return _vector3(pose.get("position"))


def vector_alignment_metrics(displacement: Any, object_direction: Any) -> dict[str, Any]:
    """Return projection/cosine metrics for an EEF displacement vector.

    ``object_direction`` is the vector from the previous EEF position to the selected
    object. Null metrics are used when either vector has zero magnitude.
    """

    displacement_vec = _vector3(displacement)
    direction_vec = _vector3(object_direction)
    displacement_norm = (
        None if displacement_vec is None else float(np.linalg.norm(displacement_vec))
    )
    direction_norm = None if direction_vec is None else float(np.linalg.norm(direction_vec))

    metrics: dict[str, Any] = {
        "displacement_norm": displacement_norm,
        "object_direction_norm": direction_norm,
        "projection_m": None,
        "cosine": None,
        "aligned_toward_object": None,
    }
    if (
        displacement_vec is None
        or direction_vec is None
        or displacement_norm is None
        or direction_norm is None
        or displacement_norm <= 1e-12
        or direction_norm <= 1e-12
    ):
        return metrics

    unit_direction = direction_vec / direction_norm
    projection = float(np.dot(displacement_vec, unit_direction))
    cosine = float(projection / displacement_norm)
    metrics.update(
        {
            "projection_m": projection,
            "cosine": float(np.clip(cosine, -1.0, 1.0)),
            "aligned_toward_object": bool(projection > 0.0),
        }
    )
    return metrics


def _distance_trend(delta: float | None, *, eps: float = 1e-9) -> str:
    if delta is None:
        return "unknown"
    if delta < -eps:
        return "decreased"
    if delta > eps:
        return "increased"
    return "unchanged"


def _nearest_distance_name(distances: dict[str, Any]) -> str | None:
    best_name = None
    best_distance = math.inf
    for name, value in (distances or {}).items():
        number = finite_float(value)
        if number is not None and number < best_distance:
            best_name = str(name)
            best_distance = number
    return best_name


def _select_object_for_arm(
    previous_gripper_pos: np.ndarray | None,
    current_gripper_pos: np.ndarray | None,
    object_poses: dict[str, Any],
    distances: dict[str, Any],
    *,
    intended_object: str | None = None,
) -> str | None:
    """Select the object whose direction best explains the measured EEF motion.

    If the EEF did not move or poses are missing, fall back to the closest reported
    object distance for that gripper.
    """

    if intended_object and intended_object in (object_poses or {}):
        return intended_object

    fallback = _nearest_distance_name(distances)
    if previous_gripper_pos is None or current_gripper_pos is None:
        return fallback

    displacement = current_gripper_pos - previous_gripper_pos
    displacement_norm = float(np.linalg.norm(displacement))
    if displacement_norm <= 1e-12:
        return fallback

    best_name = None
    best_abs_cosine = -math.inf
    for object_name, pose in (object_poses or {}).items():
        if not isinstance(pose, dict):
            continue
        object_pos = _vector3(pose.get("position"))
        if object_pos is None:
            continue
        direction = object_pos - previous_gripper_pos
        metrics = vector_alignment_metrics(displacement, direction)
        cosine = metrics.get("cosine")
        if cosine is None:
            continue
        score = abs(float(cosine))
        if score > best_abs_cosine:
            best_name = str(object_name)
            best_abs_cosine = score
    return best_name or fallback


def build_arm_probe_step(
    previous: dict[str, Any],
    current: dict[str, Any],
    intended_objects: dict[str, str] | None = None,
) -> dict[str, Any]:
    previous_min = finite_float((previous or {}).get("min_distance_m"))
    current_min = finite_float((current or {}).get("min_distance_m"))
    delta = None if previous_min is None or current_min is None else current_min - previous_min
    object_poses = (current or {}).get("object_poses") or (previous or {}).get("object_poses") or {}
    current_distances = (current or {}).get("gripper_object_distances") or {}
    previous_grippers = (previous or {}).get("gripper_poses") or {}
    current_grippers = (current or {}).get("gripper_poses") or {}

    arms: dict[str, Any] = {}
    gripper_names = sorted(set(previous_grippers) | set(current_grippers) | set(current_distances))
    for gripper_name in gripper_names:
        previous_pos = _pose_position(previous_grippers, gripper_name)
        current_pos = _pose_position(current_grippers, gripper_name)
        distances = (
            current_distances.get(gripper_name, {}) if isinstance(current_distances, dict) else {}
        )
        intended_object = (intended_objects or {}).get(gripper_name)
        selected_object = _select_object_for_arm(
            previous_pos,
            current_pos,
            object_poses,
            distances,
            intended_object=intended_object,
        )
        object_pos = (
            _pose_position(object_poses, selected_object) if selected_object is not None else None
        )

        displacement = (
            None if previous_pos is None or current_pos is None else current_pos - previous_pos
        )
        direction = (
            None if previous_pos is None or object_pos is None else object_pos - previous_pos
        )
        metrics = vector_alignment_metrics(displacement, direction)
        selected_distance = None
        if isinstance(distances, dict) and selected_object is not None:
            selected_distance = finite_float(distances.get(selected_object))
        arms[gripper_name] = {
            "selected_object": selected_object,
            "selection_method": (
                "intended_object"
                if intended_object and selected_object == intended_object
                else "alignment_or_nearest"
            ),
            "selected_distance_m": selected_distance,
            "intended_object": intended_object,
            "displacement_xyz_m": None
            if displacement is None
            else displacement.astype(float).tolist(),
            "object_direction_xyz_m": None
            if direction is None
            else direction.astype(float).tolist(),
            **metrics,
        }

    return {
        "min_distance_m": current_min,
        "previous_min_distance_m": previous_min,
        "distance_delta_m": delta,
        "distance_trend": _distance_trend(delta),
        "arms": arms,
    }


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _sign_label(value: Any, *, eps: float = 1e-9) -> str | None:
    number = finite_float(value)
    if number is None:
        return None
    if number > eps:
        return "pos"
    if number < -eps:
        return "neg"
    return "zero"


def right_translation_axis_sign_probe(row: dict[str, Any]) -> dict[str, Any]:
    transformed = np.asarray(row.get("transformed_action") or [], dtype=np.float64).reshape(-1)
    if transformed.size < RIGHT_ARM_SLICE.start + 3:
        return {}
    right_delta = transformed[RIGHT_ARM_SLICE][:3].astype(float)
    right_arm = (row.get("arms") or {}).get("right_gripper") or {}
    displacement = _vector3(right_arm.get("displacement_xyz_m"))
    out: dict[str, Any] = {}
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        action_sign = _sign_label(right_delta[axis_index])
        observed_sign = _sign_label(None if displacement is None else displacement[axis_index])
        relation = "unknown"
        if action_sign == "zero" or observed_sign == "zero":
            relation = "zero"
        elif action_sign is not None and observed_sign is not None:
            relation = "same" if action_sign == observed_sign else "opp"
        out[axis_name] = {
            "action_delta": float(right_delta[axis_index]),
            "action_sign": action_sign,
            "observed_displacement": None
            if displacement is None
            else float(displacement[axis_index]),
            "observed_sign": observed_sign,
            "relation": relation,
        }
    return out


def _apply_remapped_right_translation(
    raw_action: Any,
    base_state14: Any,
    *,
    action_scale: float,
    right_j3_offset: float,
    right_translation_signs: tuple[float, float, float],
    right_translation_order: tuple[int, int, int] = (0, 1, 2),
) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float32).reshape(14)
    base = np.asarray(base_state14, dtype=np.float32).reshape(14)
    out = raw.copy()
    for arm_slice in (LEFT_ARM_SLICE, RIGHT_ARM_SLICE):
        delta = np.clip(raw[arm_slice] - base[arm_slice], -MAX_ABS_ARM_DELTA, MAX_ABS_ARM_DELTA)
        scaled = np.clip(delta * float(action_scale), -MAX_ABS_ARM_DELTA, MAX_ABS_ARM_DELTA)
        if arm_slice == RIGHT_ARM_SLICE:
            translation = scaled[:3].copy()
            remapped = np.asarray(
                [translation[idx] for idx in right_translation_order], dtype=np.float32
            )
            remapped *= np.asarray(right_translation_signs, dtype=np.float32)
            scaled[:3] = remapped
        out[arm_slice] = base[arm_slice] + scaled
    out[RIGHT_J3_ACTION_INDEX] = base[RIGHT_J3_ACTION_INDEX] + float(right_j3_offset)
    return out.astype(np.float32)


def apply_named_adapter_option(
    raw_action: Any,
    base_state14: Any,
    *,
    action_scale: float,
    right_j3_offset: float,
    adapter_option: str,
) -> np.ndarray:
    """Apply a named temporary diagnostic adapter option.

    ``right_j2_delta_negated`` matches the current best bounded axis/sign probe:
    scale arm deltas, negate right-arm local delta dimension 2, then force the
    existing right_j3 offset. This is diagnostic-only and does not change source
    adapter semantics.
    """

    if adapter_option == "passthrough":
        return np.asarray(raw_action, dtype=np.float32).reshape(14).astype(np.float32)
    if adapter_option == "passthrough_gripflip":
        out = np.asarray(raw_action, dtype=np.float32).reshape(14).astype(np.float32).copy()
        # Flip gripper semantic: raw {0=closed,1=open} <-> {0=open,1=closed}
        out[6] = 1.0 - out[6]
        out[13] = 1.0 - out[13]
        return out
    if adapter_option == "scale4_rightj3m008":
        return apply_corrected_adapter_candidates(
            raw_action,
            base_state14,
            action_scale=action_scale,
            right_j3_offset=right_j3_offset,
        )
    if adapter_option == "right_yz_both_negated":
        return _apply_remapped_right_translation(
            raw_action,
            base_state14,
            action_scale=action_scale,
            right_j3_offset=right_j3_offset,
            right_translation_signs=(1.0, -1.0, -1.0),
        )
    if adapter_option == "right_yz_swapped":
        return _apply_remapped_right_translation(
            raw_action,
            base_state14,
            action_scale=action_scale,
            right_j3_offset=right_j3_offset,
            right_translation_signs=(1.0, 1.0, 1.0),
            right_translation_order=(0, 2, 1),
        )
    if adapter_option == "right_yz_swapped_both_negated":
        return _apply_remapped_right_translation(
            raw_action,
            base_state14,
            action_scale=action_scale,
            right_j3_offset=right_j3_offset,
            right_translation_signs=(1.0, -1.0, -1.0),
            right_translation_order=(0, 2, 1),
        )
    if adapter_option not in ("right_j2_delta_negated", "bilateral_j2_delta_negated"):
        raise ValueError(f"Unknown adapter option: {adapter_option}")

    negate_left_dim2 = adapter_option == "bilateral_j2_delta_negated"

    raw = np.asarray(raw_action, dtype=np.float32).reshape(14)
    base = np.asarray(base_state14, dtype=np.float32).reshape(14)
    out = raw.copy()
    for arm_slice in (LEFT_ARM_SLICE, RIGHT_ARM_SLICE):
        delta = np.clip(raw[arm_slice] - base[arm_slice], -MAX_ABS_ARM_DELTA, MAX_ABS_ARM_DELTA)
        signs = np.ones(6, dtype=np.float32)
        if arm_slice == RIGHT_ARM_SLICE:
            signs[2] = -1.0
        if arm_slice == LEFT_ARM_SLICE and negate_left_dim2:
            signs[2] = -1.0
        out[arm_slice] = base[arm_slice] + np.clip(
            delta * signs * float(action_scale),
            -MAX_ABS_ARM_DELTA,
            MAX_ABS_ARM_DELTA,
        )
    out[RIGHT_J3_ACTION_INDEX] = base[RIGHT_J3_ACTION_INDEX] + float(right_j3_offset)
    return out.astype(np.float32)


def _as_rgb_frame(value: Any) -> np.ndarray | None:
    try:
        frame = np.asarray(value)
    except Exception:
        return None
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        return None
    if frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        return None
    if np.issubdtype(frame.dtype, np.floating):
        max_value = float(np.nanmax(frame)) if frame.size else 1.0
        if max_value <= 1.5:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _first_observation_dict(observation: Any) -> dict[str, Any] | None:
    if isinstance(observation, list | tuple):
        if not observation:
            return None
        observation = observation[0]
    return observation if isinstance(observation, dict) else None


def _append_video_frames(observation: Any, frames: dict[str, list[np.ndarray]]) -> None:
    observation_dict = _first_observation_dict(observation)
    if observation_dict is None:
        return
    for camera_name in ("top_cam", "left_cam", "right_cam"):
        if camera_name not in observation_dict:
            continue
        frame = _as_rgb_frame(observation_dict[camera_name])
        if frame is not None:
            frames.setdefault(camera_name, []).append(frame)


def _pad_frame_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    if frame.shape[0] == target_height:
        return frame
    pad_total = target_height - frame.shape[0]
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    return np.pad(frame, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="constant")


def save_probe_videos(
    out_dir: Path,
    frames: dict[str, list[np.ndarray]],
    *,
    fps: float,
) -> dict[str, Any]:
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"video_dir": str(video_dir), "video_paths": {}, "errors": {}}
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        result["errors"]["import_imageio"] = repr(exc)
        return result

    for camera_name, camera_frames in sorted(frames.items()):
        if not camera_frames:
            continue
        path = video_dir / f"{camera_name}.mp4"
        try:
            imageio.mimsave(str(path), camera_frames, fps=float(fps), macro_block_size=1)
            result["video_paths"][camera_name] = str(path)
        except Exception as exc:
            result["errors"][camera_name] = repr(exc)

    combined_names = [name for name in ("top_cam", "left_cam", "right_cam") if frames.get(name)]
    if len(combined_names) >= 2:
        combined_path = video_dir / "combined_top_left_right.mp4"
        try:
            frame_count = min(len(frames[name]) for name in combined_names)
            combined_frames = []
            for idx in range(frame_count):
                row = [frames[name][idx] for name in combined_names]
                height = max(frame.shape[0] for frame in row)
                row = [_pad_frame_to_height(frame, height) for frame in row]
                combined_frames.append(np.hstack(row))
            imageio.mimsave(str(combined_path), combined_frames, fps=float(fps), macro_block_size=1)
            result["video_paths"]["combined"] = str(combined_path)
        except Exception as exc:
            result["errors"]["combined"] = repr(exc)
    result["frame_counts"] = {name: len(value) for name, value in sorted(frames.items())}
    return result


def aggregate_alignment_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {
        "distance_trends": {
            "decreased": 0,
            "increased": 0,
            "unchanged": 0,
            "unknown": 0,
        },
        "right_translation_axis_relation_counts": {
            axis: {"same": 0, "opp": 0, "zero": 0, "unknown": 0} for axis in ("x", "y", "z")
        },
    }
    arm_values: dict[str, dict[str, Any]] = {}
    for row in rows:
        for axis_name, axis_metrics in (row.get("right_translation_axis_probe") or {}).items():
            relation = axis_metrics.get("relation", "unknown")
            if relation not in aggregates["right_translation_axis_relation_counts"][axis_name]:
                relation = "unknown"
            aggregates["right_translation_axis_relation_counts"][axis_name][relation] += 1
        trend = row.get("distance_trend", "unknown")
        if trend not in aggregates["distance_trends"]:
            trend = "unknown"
        aggregates["distance_trends"][trend] += 1
        for arm_name, metrics in (row.get("arms") or {}).items():
            bucket = arm_values.setdefault(
                arm_name,
                {
                    "measured_steps": 0,
                    "missing_steps": 0,
                    "aligned_steps": 0,
                    "displacement_norms": [],
                    "projections": [],
                    "cosines": [],
                },
            )
            if metrics.get("cosine") is None or metrics.get("projection_m") is None:
                bucket["missing_steps"] += 1
                continue
            bucket["measured_steps"] += 1
            if metrics.get("aligned_toward_object") is True:
                bucket["aligned_steps"] += 1
            displacement = finite_float(metrics.get("displacement_norm"))
            projection = finite_float(metrics.get("projection_m"))
            cosine = finite_float(metrics.get("cosine"))
            if displacement is not None:
                bucket["displacement_norms"].append(displacement)
            if projection is not None:
                bucket["projections"].append(projection)
            if cosine is not None:
                bucket["cosines"].append(cosine)

    for arm_name, bucket in arm_values.items():
        measured = int(bucket["measured_steps"])
        aggregates[arm_name] = {
            "measured_steps": measured,
            "missing_steps": int(bucket["missing_steps"]),
            "aligned_steps": int(bucket["aligned_steps"]),
            "aligned_fraction": None
            if measured == 0
            else float(bucket["aligned_steps"] / measured),
            "mean_cosine": _mean_or_none(bucket["cosines"]),
            "mean_projection_m": _mean_or_none(bucket["projections"]),
            "mean_displacement_norm_m": _mean_or_none(bucket["displacement_norms"]),
        }
    return aggregates


def write_artifacts(out_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    report["report_path"] = str(out_dir / "report.json")
    report["per_step_path"] = str(out_dir / "per_step.jsonl")
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "per_step.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _norm_delta(action: np.ndarray, base_state14: np.ndarray, arm_slice: slice) -> float:
    return float(np.linalg.norm(action[arm_slice] - base_state14[arm_slice]))


def _flat_dict(values: dict[str, Any] | None) -> dict[str, list[float]] | None:
    if values is None:
        return None
    return {str(name): flat(value) for name, value in sorted(values.items())}


def capture_robot_control_snapshot(task: Any) -> dict[str, Any]:
    """Capture commanded, controller-target, and actual joint positions after one step."""

    robot = task._env.robots[0]
    controller_targets = {
        name: controller.target_pos.copy()
        for name, controller in robot.controllers.items()
        if hasattr(controller, "target_pos")
    }
    return {
        "unnoised_command_target": _flat_dict(robot.last_unnoised_cmd_joint_pos()),
        "controller_target": _flat_dict(controller_targets),
        "actual_qpos": _flat_dict(robot.robot_view.get_qpos_dict()),
    }


def max_group_abs_error(
    lhs: dict[str, list[float]] | None,
    rhs: dict[str, list[float]] | None,
) -> float | None:
    if not lhs or not rhs:
        return None
    errors: list[float] = []
    for name in sorted(set(lhs) & set(rhs)):
        left = np.asarray(lhs[name], dtype=np.float64)
        right = np.asarray(rhs[name], dtype=np.float64)
        if left.shape == right.shape and left.size:
            errors.append(float(np.max(np.abs(left - right))))
    return max(errors) if errors else None


def _initial_report(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    return {
        "status": "started",
        "evidence_label": FRAME_PROBE_LABEL,
        "related_evidence_label": DIAGNOSTIC_EVIDENCE_LABEL,
        "artifact_dir": str(out_dir),
        "endpoint": args.endpoint,
        "constraints": {
            "max_executed_steps": int(args.max_steps),
            "no_long_rollout": True,
            "act_service_restart_requested": False,
            "act_service_stop_requested": False,
            "source_adapter_semantics_modified": False,
        },
        "baseline_contract": {
            "seed": 0,
            "endpoint": OFFICIAL_BASELINE_ENDPOINT,
            "action_noise_enabled": False,
            "execution_command_hz": OFFICIAL_BASELINE_COMMAND_HZ,
            "chunk_consumption": "complete",
        },
        "candidate_parameters": {
            "action_scale": float(args.action_scale),
            "right_j3_offset": float(args.right_j3_offset),
            "right_j3_action_index": RIGHT_J3_ACTION_INDEX,
            "right_j3_convention": "zero-based joint index 3 within right arm, action index 10",
            "right_j2_delta_negated_action_index": int(RIGHT_ARM_SLICE.start + 2),
            "adapter_option": args.adapter_option,
            "max_abs_arm_delta": MAX_ABS_ARM_DELTA,
        },
        "intended_objects": {
            "left_gripper": args.intended_left_object,
            "right_gripper": args.intended_right_object,
        },
        "act_response_present": False,
        "selected_action_shape": None,
        "selected_action_shape_before_limit": None,
        "executed_steps": 0,
        "first_distance_m": None,
        "min_distance_m": None,
        "last_distance_m": None,
        "distance_trace_m": [],
        "alignment_metrics": {},
        "controller_qpos_clipping_or_saturation": None,
        "contact_flags": {"any_gripper_contact": False, "latest_any_gripper_contact": False},
        "final_n_in_box": None,
        "success": None,
        "raw_action_log_path": str(out_dir / "raw_act_response.jsonl"),
        "debug_dump_dir": str(out_dir / "debug_request"),
        "video": {
            "required_for_complete_run": True,
            "fps": float(args.video_fps),
            "video_dir": str(out_dir / "videos"),
            "video_paths": {},
            "errors": {},
        },
        "teardown_note": (
            "The diagnostic attaches to the existing /act service and does not start, stop, "
            "or restart it. os._exit is used by default after final flush because sampler "
            "close/env cleanup has historically hung on this host."
        ),
    }


def run_probe(args: argparse.Namespace) -> int:
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

    out_dir = make_artifact_dir(args.artifact_root, stamp=args.timestamp)
    rows: list[dict[str, Any]] = []
    video_frames: dict[str, list[np.ndarray]] = {}
    report = _initial_report(args, out_dir)
    write_artifacts(out_dir, report, rows)

    sampler = None
    try:
        cfg = MolmoAct2OfficialSimEvalYamBoxDataGenConfig()
        cfg.task_horizon = max(int(args.max_steps) + 2, 8)
        cfg.policy_config.remote_config = {
            "host": "127.0.0.1",
            "port": 8203,
            "path": "/act",
            "timeout": float(args.timeout),
        }
        cfg.policy_config.endpoint_url = args.endpoint
        cfg.policy_config.timeout = float(args.timeout)
        cfg.policy_config.num_steps = int(args.max_steps)
        cfg.policy_config.n_action_steps = None
        cfg.policy_config.execution_mode = "sim_eval_step"
        cfg.policy_config.execution_command_hz = OFFICIAL_BASELINE_COMMAND_HZ
        cfg.policy_config.raw_action_log_path = str(out_dir / "raw_act_response.jsonl")
        cfg.policy_config.debug_dump_dir = str(out_dir / "debug_request")
        cfg.policy_config.debug_dump_max_calls = int(args.debug_dump_max_calls)

        sampler = cfg.task_sampler_config.task_sampler_class(cfg)
        task = sampler.sample_task(force_advance_scene=True, house_index=0)
        policy = MolmoAct2YamPolicy(cfg, task=task)

        replan_every = int(args.replan_every)

        def _fresh_request_for_report() -> tuple[dict[str, Any], np.ndarray]:
            fresh_obs = task.get_observations()
            _append_video_frames(fresh_obs, video_frames)
            fresh_request = policy.obs_to_model_input(fresh_obs)
            _append_video_frames(fresh_request, video_frames)
            fresh_base = np.asarray(fresh_request["state"], dtype=np.float32).reshape(14)
            return fresh_request, fresh_base

        request, base_state14 = _fresh_request_for_report()
        report["act_response_present"] = True
        report["endpoint"] = policy.endpoint_url
        report["request_num_steps"] = request.get("num_steps")
        report["selected_action_shape_before_limit"] = (
            list(np.asarray(policy.actions_buffer).shape)
            if policy.actions_buffer is not None
            else None
        )
        report["selected_action_shape"] = report["selected_action_shape_before_limit"]
        report["replan_events"] = []
        previous_summary = summarize_info(task.get_info())
        last_executed_grippers: dict[str, float] | None = None
        chunk_id = -1
        write_artifacts(out_dir, report, rows)

        max_total_steps = int(args.max_steps)
        step_idx = 0
        while step_idx < max_total_steps:
            step_obs = task.get_observations()
            _append_video_frames(step_obs, video_frames)
            step_request = policy.obs_to_model_input(step_obs)
            _append_video_frames(step_request, video_frames)
            base_state14 = np.asarray(step_request["state"], dtype=np.float32).reshape(14)

            if replan_every > 0 and step_idx > 0 and step_idx % replan_every == 0:
                policy.actions_buffer = None
                policy.current_buffer_index = 0
                report["replan_events"].append(
                    {"step": step_idx, "reason": "forced_replan_every_n"}
                )

            raw, chunk_id, chunk_index, new_chunk = infer_buffered_action_with_chunk_metadata(
                policy,
                step_request,
                chunk_id=chunk_id,
            )
            if new_chunk:
                report["replan_events"].append(
                    {
                        "step": step_idx,
                        "reason": "policy_chunk_boundary",
                        "chunk_id": chunk_id,
                    }
                )
            step_idx += 1
            transformed = apply_named_adapter_option(
                raw,
                base_state14,
                action_scale=float(args.action_scale),
                right_j3_offset=float(args.right_j3_offset),
                adapter_option=str(args.adapter_option),
            )
            transport_active = bool(
                previous_summary.get("any_gripper_contact", False)
                or previous_summary.get("held", False)
            )
            transformed, guard = apply_gripper_continuity_guard(
                transformed,
                chunk_index=chunk_index,
                guard_steps=(int(args.gripper_continuity_guard_steps) if chunk_id > 0 else 0),
                transport_active=transport_active,
                last_executed_grippers=last_executed_grippers,
            )
            command = policy.model_output_to_action(transformed)
            command_target = _flat_dict(command)
            _observation, reward, terminated, truncated, info = execute_molmoact2_yam_action(
                task,
                command,
                execution_mode=cfg.policy_config.execution_mode,
                joint_step=0.02,
                max_smoothing_steps=3,
                command_hz=cfg.policy_config.execution_command_hz,
            )
            post_step_observations = task.get_observations()
            _append_video_frames(post_step_observations, video_frames)
            actual_normalized_state14 = np.asarray(
                policy.obs_to_model_input(post_step_observations)["state"],
                dtype=np.float32,
            ).reshape(14)
            control_snapshot = capture_robot_control_snapshot(task)
            control_snapshot["policy_command_target"] = command_target
            control_snapshot["max_abs_unnoised_to_controller_target"] = max_group_abs_error(
                control_snapshot["unnoised_command_target"],
                control_snapshot["controller_target"],
            )
            control_snapshot["max_abs_controller_target_to_actual_qpos"] = max_group_abs_error(
                control_snapshot["controller_target"],
                control_snapshot["actual_qpos"],
            )
            current_summary = summarize_info(info)
            intended_objects = {
                "left_gripper": str(args.intended_left_object),
                "right_gripper": str(args.intended_right_object),
            }
            probe_row = build_arm_probe_step(previous_summary, current_summary, intended_objects)
            probe_row.update(
                {
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
                    "any_gripper_contact": current_summary["any_gripper_contact"],
                    "n_in_box": current_summary["n_in_box"],
                    "success": current_summary["success"],
                    "distances": current_summary["gripper_object_distances"],
                    "contacts": current_summary["contacts"],
                    "gripper_poses": current_summary["gripper_poses"],
                    "object_poses": current_summary["object_poses"],
                    "actual_normalized_state": flat(actual_normalized_state14),
                    "control_snapshot": control_snapshot,
                    "controller_qpos_clipping_or_saturation": control_snapshot[
                        "max_abs_unnoised_to_controller_target"
                    ],
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "gripper_continuity_guard": guard,
                }
            )
            if args.structured_step_diagnostics:
                probe_row["structured_step_diagnostic"] = build_structured_step_diagnostic(
                    step=step_idx,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    raw_action=raw,
                    mapped_action=transformed,
                    actual_state14=actual_normalized_state14,
                    summary=current_summary,
                    previous_summary=previous_summary,
                    guard=guard,
                    force_diagnostics=collect_optional_mujoco_forces(
                        task,
                        requested=bool(args.include_mujoco_forces),
                    ),
                )
                probe_row["structured_step_diagnostic"]["state_semantics"] = {
                    "actual_normalized_state": "model-input normalized 14D state",
                    "actual_normalized_opening": (
                        "normalized gripper state (0=closed, 1=open), not meters"
                    ),
                }
            probe_row["right_translation_axis_probe"] = right_translation_axis_sign_probe(probe_row)
            rows.append(probe_row)
            distance = current_summary["min_distance_m"]
            report["distance_trace_m"].append(distance)
            valid_distances = [d for d in report["distance_trace_m"] if d is not None]
            report["first_distance_m"] = valid_distances[0] if valid_distances else None
            report["min_distance_m"] = min(valid_distances) if valid_distances else None
            report["last_distance_m"] = valid_distances[-1] if valid_distances else None
            report["executed_steps"] = step_idx
            report["alignment_metrics"] = aggregate_alignment_metrics(rows)
            report["contact_flags"] = {
                "any_gripper_contact": bool(
                    report["contact_flags"]["any_gripper_contact"]
                    or current_summary["any_gripper_contact"]
                ),
                "latest_any_gripper_contact": bool(current_summary["any_gripper_contact"]),
            }
            report["final_n_in_box"] = current_summary["n_in_box"]
            report["success"] = bool(current_summary["success"])
            report["status"] = "running"
            write_artifacts(out_dir, report, rows)
            last_executed_grippers = {
                "left": float(transformed[6]),
                "right": float(transformed[13]),
            }
            previous_summary = current_summary

            if (
                current_summary["success"]
                or bool(np.asarray(terminated).any())
                or bool(np.asarray(truncated).any())
                or step_idx >= int(args.max_steps)
            ):
                break

        report["video"] = save_probe_videos(out_dir, video_frames, fps=float(args.video_fps))
        report["status"] = "completed"
        report["finished_at_epoch"] = time.time()
        report["alignment_metrics"] = aggregate_alignment_metrics(rows)
        if not report.get("video", {}).get("video_paths"):
            report["status"] = "completed_without_video"
        write_artifacts(out_dir, report, rows)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "artifact_dir": str(out_dir),
                    "executed_steps": report["executed_steps"],
                    "selected_action_shape": report["selected_action_shape"],
                    "min_distance_m": report["min_distance_m"],
                    "last_distance_m": report["last_distance_m"],
                    "contact_flags": report["contact_flags"],
                    "final_n_in_box": report["final_n_in_box"],
                    "success": report["success"],
                    "alignment_metrics": report["alignment_metrics"],
                    "video": report.get("video"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        report["video"] = save_probe_videos(out_dir, video_frames, fps=float(args.video_fps))
        report["status"] = "failed"
        report["finished_at_epoch"] = time.time()
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        write_artifacts(out_dir, report, rows)
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
    code = run_probe(args)
    if args.no_os_exit:
        return code
    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
