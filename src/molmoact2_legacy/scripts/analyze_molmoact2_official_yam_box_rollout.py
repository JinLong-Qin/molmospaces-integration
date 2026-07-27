"""Analyze a MolmoAct2 official YAM box rollout and write evidence artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from molmo_spaces.data_generation.config.molmoact2_official_yam_box_config import (
    OFFICIAL_MOLMOACT2_YAM_ENV_ID,
    OFFICIAL_MOLMOACT2_YAM_SOURCE_ROBOT,
    OFFICIAL_MOLMOACT2_YAM_SOURCE_TASK,
    OFFICIAL_YAM_BALL_NAME,
    OFFICIAL_YAM_LEGO_NAME,
    OFFICIAL_YAM_OBJECT_NAMES,
    OFFICIAL_YAM_OPEN_BOX_NAME,
)
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
    MOLMOACT2_YAM_ACTION_GRIPPER_SEMANTICS,
)

ARTIFACT_DIR = Path("artifacts/molmospaces/molmoact2_yam_integration")
DEFAULT_RUN_PREFIX = "molmoact2_official_yam_box_400step_20260623"
DEFAULT_RAW_ACTION_LOG = Path(
    "artifacts/molmospaces/molmoact2_official_yam_box_debug/raw_actions.jsonl"
)
TASK_ROOT = (
    Path("/home/c/project/paper_reproductions/official/artifacts")
    / "molmospaces/molmoact2_yam_integration_analysis_20260620_142443/molmoact2"
)
OFFICIAL_SOURCE_PATHS = {
    "task": str(TASK_ROOT / OFFICIAL_MOLMOACT2_YAM_SOURCE_TASK),
    "robot": str(TASK_ROOT / OFFICIAL_MOLMOACT2_YAM_SOURCE_ROBOT),
}
CONTACT_DISTANCE_THRESHOLD_M = 0.06
APPROACH_DELTA_THRESHOLD_M = 0.02
LIFT_DELTA_THRESHOLD_M = 0.03


def decode_json_byte_array(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.kind in {"S", "O"} and array.shape == ():
        raw = array[()].decode("utf-8") if isinstance(array[()], bytes) else str(array[()])
    else:
        raw = bytes(array.astype(np.uint8)).split(b"\x00", 1)[0].decode("utf-8")
    return json.loads(raw) if raw else {}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def find_latest_run_dir(run_prefix: str) -> Path | None:
    datagen_root = ASSETS_DIR / "datagen" / "packing_molmoact2_yam_v1"
    if not datagen_root.exists():
        return None
    candidates = [
        path
        for path in datagen_root.iterdir()
        if path.is_dir() and path.name.startswith(f"{run_prefix}_")
    ]
    if not candidates:
        candidates = [
            path
            for path in datagen_root.iterdir()
            if path.is_dir() and path.name.startswith(run_prefix)
        ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def find_h5(run_dir: Path) -> Path | None:
    h5_paths = sorted(run_dir.glob("house_*/trajectories*.h5"))
    return h5_paths[0] if h5_paths else None


def read_dataset(group: h5py.Group, key: str) -> np.ndarray | None:
    if key not in group:
        return None
    return np.asarray(group[key])


def summarize_array(array: np.ndarray | None) -> dict[str, Any] | None:
    if array is None:
        return None
    summary: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size:
        summary["last"] = to_jsonable(array.reshape(-1)[-1])
        if np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_:
            summary["sum"] = float(np.asarray(array, dtype=float).sum())
    return summary


def extract_task_info(traj: h5py.Group) -> tuple[list[dict[str, Any]], str | None]:
    task_info_path = "obs/extra/task_info"
    if task_info_path not in traj:
        return [], f"H5 missing {task_info_path}; per-step task_info behavior log unavailable."
    rows = []
    for row in traj[task_info_path]:
        try:
            rows.append(decode_json_byte_array(row))
        except Exception as exc:  # noqa: BLE001
            return rows, f"Failed to decode {task_info_path}: {exc}"
    return rows, None


def extract_commanded_actions(traj: h5py.Group) -> list[dict[str, Any]]:
    action_path = "actions/commanded_action"
    if action_path not in traj:
        return []
    actions = []
    for row in traj[action_path]:
        try:
            actions.append(decode_json_byte_array(row))
        except Exception:
            actions.append({})
    return actions


def flatten_numeric_dict_values(value: dict[str, Any]) -> np.ndarray:
    numbers: list[float] = []
    for key in sorted(value):
        item = value[key]
        if isinstance(item, dict):
            numbers.extend(flatten_numeric_dict_values(item).tolist())
        else:
            try:
                array = np.asarray(item, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            numbers.extend(array.tolist())
    return np.asarray(numbers, dtype=float)


def extract_qpos_vectors(traj: h5py.Group) -> list[np.ndarray]:
    qpos_path = "obs/agent/qpos"
    if qpos_path not in traj:
        return []
    vectors = []
    for row in traj[qpos_path]:
        try:
            vectors.append(flatten_numeric_dict_values(decode_json_byte_array(row)))
        except Exception:
            vectors.append(np.asarray(row, dtype=float).reshape(-1))
    return vectors


def read_obs_scene(traj: h5py.Group) -> dict[str, Any]:
    if "obs_scene" not in traj:
        return {}
    value = traj["obs_scene"][()]
    if isinstance(value, bytes):
        return json.loads(value.decode("utf-8"))
    return json.loads(str(value))


def task_info_object_position(info: dict[str, Any], object_name: str) -> list[float] | None:
    behavior = info.get("official_yam_behavior", {})
    pose = behavior.get("object_poses", {}).get(object_name)
    if pose and "position" in pose:
        return pose["position"]
    per_object = info.get("official_yam_box", {}).get("per_object", {}).get(object_name)
    if per_object and "position" in per_object:
        return per_object["position"]
    return None


def compact_action(action: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {"keys": sorted(action.keys())}
    for key, value in action.items():
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            compact[key] = {"shape": list(arr.shape)}
        else:
            compact[key] = {
                "shape": list(arr.shape),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }
    return compact


def load_raw_action_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def write_raw_action_segment(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(to_jsonable(entry), sort_keys=True) + "\n")


def infer_raw_action_slice(
    entries: list[dict[str, Any]],
    run_started_at: float | None,
) -> list[dict[str, Any]]:
    if not entries:
        return []
    last_segment_start = None
    for index, entry in enumerate(entries):
        if int(entry.get("call_index", -1)) == 0:
            last_segment_start = index
    if last_segment_start is not None:
        return entries[last_segment_start:]
    if run_started_at is None:
        return entries
    recent = [
        entry for entry in entries if float(entry.get("timestamp", 0.0)) >= run_started_at - 5.0
    ]
    return recent or entries


def collect_video_paths(h5_path: Path) -> dict[str, str]:
    house_dir = h5_path.parent
    paths = {}
    for camera in ("top_cam", "left_cam", "right_cam"):
        matches = sorted(house_dir.glob(f"episode_*_{camera}_batch_*.mp4"))
        if matches:
            paths[camera] = str(matches[0])
    return paths


def create_side_by_side_video(video_paths: dict[str, str], output_path: Path) -> str | None:
    if len(video_paths) != 3 or shutil.which("ffmpeg") is None:
        return None
    inputs = []
    for camera in ("top_cam", "left_cam", "right_cam"):
        if camera not in video_paths:
            return None
        inputs.extend(["-i", video_paths[camera]])
    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        "[0:v][1:v][2:v]hstack=inputs=3[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        str(output_path),
    ]
    result = subprocess.run(
        cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        return None
    return str(output_path)


def probe_act_server() -> dict[str, Any]:
    import requests

    url = "http://127.0.0.1:8202/act"
    proxies = {"http": None, "https": None}
    try:
        response = requests.get(url, timeout=5.0, proxies=proxies)
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = response.text[:500]
        return {"url": url, "get_status_http": response.status_code, "body": body}
    except Exception as exc:  # noqa: BLE001
        return {"url": url, "error": repr(exc)}


def run_started_at_from_dir(run_dir: Path | None) -> float | None:
    if run_dir is None or not run_dir.exists():
        return None
    return run_dir.stat().st_ctime


def analyze_behavior(
    task_infos: list[dict[str, Any]],
    commanded_actions: list[dict[str, Any]],
    raw_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
    steps = []
    first_positions = (
        {
            object_name: task_info_object_position(task_infos[0], object_name)
            for object_name in OFFICIAL_YAM_OBJECT_NAMES
        }
        if task_infos
        else {}
    )
    min_distance = {
        object_name: {"left_gripper": math.inf, "right_gripper": math.inf}
        for object_name in OFFICIAL_YAM_OBJECT_NAMES
    }
    max_lift_delta = {object_name: 0.0 for object_name in OFFICIAL_YAM_OBJECT_NAMES}
    ever = {
        "approach": False,
        "contact": False,
        "grasp": False,
        "lift": False,
        "place": False,
    }
    event_steps: dict[str, list[int]] = {key: [] for key in ever}

    for index, info in enumerate(task_infos):
        behavior = info.get("official_yam_behavior", {})
        distances = behavior.get("gripper_object_distances", {})
        contacts = behavior.get("contacts", {})
        per_object_metrics = info.get("official_yam_box", {}).get("per_object", {})

        object_positions = {}
        for object_name in OFFICIAL_YAM_OBJECT_NAMES + (OFFICIAL_YAM_OPEN_BOX_NAME,):
            object_positions[object_name] = task_info_object_position(info, object_name)

        for gripper_name, per_object in distances.items():
            for object_name, distance in per_object.items():
                if object_name in min_distance and gripper_name in min_distance[object_name]:
                    min_distance[object_name][gripper_name] = min(
                        min_distance[object_name][gripper_name], float(distance)
                    )
                    if float(distance) <= CONTACT_DISTANCE_THRESHOLD_M:
                        ever["approach"] = True
                        event_steps["approach"].append(index)

        for object_name in OFFICIAL_YAM_OBJECT_NAMES:
            start_pos = first_positions.get(object_name)
            current_pos = object_positions.get(object_name)
            if start_pos and current_pos:
                dz = float(current_pos[2] - start_pos[2])
                max_lift_delta[object_name] = max(max_lift_delta[object_name], dz)
                if dz >= LIFT_DELTA_THRESHOLD_M:
                    ever["lift"] = True
                    event_steps["lift"].append(index)

            if per_object_metrics.get(object_name, {}).get("in_box", False):
                ever["place"] = True
                event_steps["place"].append(index)

            object_contact = contacts.get(object_name, {})
            if object_contact.get("any_gripper_touching", False):
                ever["contact"] = True
                event_steps["contact"].append(index)
            for grip_state in object_contact.get("per_gripper", {}).values():
                if grip_state.get("held", False):
                    ever["grasp"] = True
                    event_steps["grasp"].append(index)

        raw_call_index = index // 30 if raw_entries else None
        chunk_index = index % 30 if raw_entries else None
        steps.append(
            {
                "timestep": index,
                "success": bool(info.get("success", False)),
                "reward": float(info.get("n_in_box", 0)) / max(float(info.get("n_total", 1)), 1.0),
                "done": None,
                "truncated": None,
                "object_positions": object_positions,
                "gripper_poses": behavior.get("gripper_poses", {}),
                "distances": distances,
                "contacts": contacts,
                "action_summary": compact_action(commanded_actions[index])
                if index < len(commanded_actions)
                else {},
                "act_call_index": raw_call_index,
                "act_chunk_index": chunk_index,
            }
        )

    behavior_summary = {
        "min_gripper_object_distance_m": {
            object_name: {
                gripper_name: None if math.isinf(distance) else distance
                for gripper_name, distance in per_gripper.items()
            }
            for object_name, per_gripper in min_distance.items()
        },
        "max_object_lift_delta_m": max_lift_delta,
        "event_steps": {key: sorted(set(value))[:20] for key, value in event_steps.items()},
        "thresholds": {
            "approach_distance_m": CONTACT_DISTANCE_THRESHOLD_M,
            "lift_delta_m": LIFT_DELTA_THRESHOLD_M,
        },
    }
    return steps, behavior_summary, ever


def visibility_summary(traj: h5py.Group) -> dict[str, Any]:
    root_path = "obs/extra/object_image_points"
    if root_path not in traj:
        return {"available": False, "blocker": f"H5 missing {root_path}"}
    root = traj[root_path]
    summary: dict[str, Any] = {"available": True, "objects": {}}
    for object_key in ("lego_duplo", "tennis_ball", "open_box"):
        if object_key not in root:
            continue
        summary["objects"][object_key] = {}
        for camera in ("top_cam", "left_cam", "right_cam"):
            camera_path = f"{object_key}/{camera}/num_points"
            if camera_path in root:
                values = np.asarray(root[camera_path]).reshape(-1)
                summary["objects"][object_key][camera] = {
                    "max_points": int(values.max()) if values.size else 0,
                    "visible_steps": int(np.count_nonzero(values > 0)),
                }
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def status_text(pass_value: bool, pass_message: str, fail_message: str) -> str:
    return f"PASS: {pass_message}" if pass_value else f"FAIL: {fail_message}"


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    evidence = metrics["evidence"]
    behavior = metrics["behavior_flags"]
    success_metric = metrics["success_metric"]
    side_by_side = metrics.get("side_by_side_video_path")
    video_paths = metrics.get("video_paths", {})

    report = f"""# MolmoAct2 Official YAM Box 400-Step Behavior Validation

Timestamp: {metrics["timestamp"]} Asia/Shanghai

## now done
- Added scoped official-box per-step behavior fields into saved `task_info` and a post-run analyzer.
- Ran/inspected the official minimum 400-step route for `{OFFICIAL_MOLMOACT2_YAM_ENV_ID}`.
- Wrote metrics JSON, behavior JSONL, and this report under `{ARTIFACT_DIR}`.

## what it proves
- API/server: {evidence["API/server"]}.
- Sim stepping: {evidence["sim stepping"]}.
- Artifact saved: {evidence["artifact saved"]}.
- Object visibility: {evidence["object visibility"]}.
- Success metric: {evidence["success metric"]}.

## what it does not prove
- It does not prove task success unless `success[-1]` or any official success row is true.
- It does not prove normal-condition approach/contact/grasp/lift/place unless those decoded behavior flags are true.
- It does not prove ManiSkill equivalence; this remains a MolmoSpaces/MuJoCo official-route approximation using primitive scene assets.

## blocker
{metrics["blocker"]}

## next gate
{metrics["next_gate"]}

## Evidence Matrix
| Evidence layer | Result |
|---|---|
| API/server | {evidence["API/server"]} |
| sim stepping | {evidence["sim stepping"]} |
| artifact saved | {evidence["artifact saved"]} |
| object visibility | {evidence["object visibility"]} |
| approach/contact/grasp/lift/place | {evidence["approach/contact/grasp/lift/place"]} |
| success metric | {evidence["success metric"]} |
| teacher-facing result | {evidence["teacher-facing result"]} |

## Artifact Paths
- run dir: `{metrics.get("run_dir")}`
- H5: `{metrics.get("h5_path")}`
- metrics JSON: `{metrics.get("metrics_path")}`
- behavior log JSONL: `{metrics.get("behavior_log_path")}`
- raw `/act` log: `{metrics.get("raw_action_log_path")}`
- selected raw `/act` segment: `{metrics.get("raw_action_segment_path")}`
"""
    if side_by_side:
        report += f"- side-by-side three-camera MP4: `{side_by_side}`\n"
    else:
        report += "- side-by-side three-camera MP4: not produced; original camera MP4s below.\n"
    for camera, video_path in video_paths.items():
        report += f"- {camera} MP4: `{video_path}`\n"

    report += f"""
## Behavior Summary
- steps saved: {metrics.get("num_timepoints")}
- final success: {success_metric.get("last_success")}
- any success: {success_metric.get("any_success")}
- final reward: {success_metric.get("last_reward")}
- contact appeared: {behavior.get("contact")}
- grasp/held appeared: {behavior.get("grasp")}
- lift appeared: {behavior.get("lift")}
- place/in-box appeared: {behavior.get("place")}
- gripper semantics: `{MOLMOACT2_YAM_ACTION_GRIPPER_SEMANTICS}`
"""
    path.write_text(report, encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = args.timestamp or timestamp()
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir(args.run_prefix)
    h5_path = Path(args.h5) if args.h5 else (find_h5(run_dir) if run_dir else None)
    metrics_path = ARTIFACT_DIR / f"{args.run_prefix}_{ts}_metrics.json"
    behavior_log_path = ARTIFACT_DIR / f"{args.run_prefix}_{ts}_behavior_log.jsonl"
    report_path = ARTIFACT_DIR / f"{args.run_prefix}_{ts}_report.md"
    raw_action_segment_path = ARTIFACT_DIR / f"{args.run_prefix}_{ts}_raw_actions_selected.jsonl"
    side_by_side_path = ARTIFACT_DIR / f"{args.run_prefix}_{ts}_three_camera.mp4"
    raw_action_log_path = Path(args.raw_action_log)

    metrics: dict[str, Any] = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_prefix": args.run_prefix,
        "run_dir": str(run_dir) if run_dir else None,
        "h5_path": str(h5_path) if h5_path else None,
        "metrics_path": str(metrics_path),
        "behavior_log_path": str(behavior_log_path),
        "report_path": str(report_path),
        "raw_action_segment_path": str(raw_action_segment_path),
        "raw_action_log_path": str(raw_action_log_path),
        "official_env_id": OFFICIAL_MOLMOACT2_YAM_ENV_ID,
        "official_source_paths": OFFICIAL_SOURCE_PATHS,
        "act_server_probe": probe_act_server(),
    }

    if h5_path is None or not h5_path.exists():
        metrics["blocker"] = "No H5 trajectory found for the requested 400-step run."
        metrics["next_gate"] = (
            "Rerun the exact 400-step pipeline command after fixing the missing artifact."
        )
        metrics["evidence"] = {
            "API/server": "UNKNOWN: no rollout artifact was available for inspection",
            "sim stepping": "FAIL: no H5 trajectory was found",
            "artifact saved": "FAIL: no H5 trajectory was found",
            "object visibility": "UNKNOWN: no H5 trajectory was found",
            "approach/contact/grasp/lift/place": "UNKNOWN: no behavior log was available",
            "success metric": "UNKNOWN: no success array was available",
            "teacher-facing result": "FAIL: no teacher-facing successful episode evidence",
        }
        metrics_path.write_text(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True) + "\n")
        write_report(report_path, metrics)
        return metrics

    raw_entries_all = load_raw_action_log(raw_action_log_path)
    raw_entries = infer_raw_action_slice(raw_entries_all, run_started_at_from_dir(run_dir))
    write_raw_action_segment(raw_action_segment_path, raw_entries)
    video_paths = collect_video_paths(h5_path)
    side_by_side_video = create_side_by_side_video(video_paths, side_by_side_path)

    with h5py.File(h5_path, "r") as h5_file:
        traj_key = sorted(h5_file.keys())[0]
        traj = h5_file[traj_key]
        rewards = read_dataset(traj, "rewards")
        success = read_dataset(traj, "success")
        terminated = read_dataset(traj, "terminated")
        truncated = read_dataset(traj, "truncated")
        fail = read_dataset(traj, "fail")
        task_infos, task_info_blocker = extract_task_info(traj)
        commanded_actions = extract_commanded_actions(traj)
        obs_scene = read_obs_scene(traj)
        object_visibility = visibility_summary(traj)
        qpos_vectors = extract_qpos_vectors(traj)

    behavior_steps, behavior_summary, behavior_flags = analyze_behavior(
        task_infos,
        commanded_actions,
        raw_entries,
    )
    for idx, step in enumerate(behavior_steps):
        if rewards is not None and idx < len(rewards):
            step["reward"] = float(rewards[idx])
        if success is not None and idx < len(success):
            step["success"] = bool(success[idx])
        if terminated is not None and idx < len(terminated):
            step["done"] = bool(terminated[idx])
        if truncated is not None and idx < len(truncated):
            step["truncated"] = bool(truncated[idx])
    write_jsonl(behavior_log_path, behavior_steps)

    num_timepoints = int(len(success)) if success is not None else len(task_infos)
    any_success = bool(np.any(success)) if success is not None and success.size else False
    last_success = bool(success[-1]) if success is not None and success.size else False
    reward_sum = float(np.asarray(rewards, dtype=float).sum()) if rewards is not None else None
    last_reward = float(rewards[-1]) if rewards is not None and rewards.size else None
    expected_min_timepoints = 401
    ran_400_step_gate = num_timepoints >= expected_min_timepoints or (
        truncated is not None and truncated.size and bool(truncated[-1]) and num_timepoints >= 400
    )
    action_nonempty = sum(1 for action in commanded_actions if action)
    raw_act_call_count = len(raw_entries)
    qpos_delta_norm = None
    qpos_first = None
    qpos_last = None
    if len(qpos_vectors) >= 2 and qpos_vectors[0].shape == qpos_vectors[-1].shape:
        qpos_first = qpos_vectors[0]
        qpos_last = qpos_vectors[-1]
        qpos_delta_norm = float(np.linalg.norm(qpos_last - qpos_first))

    visibility_pass = False
    if object_visibility.get("available"):
        visibility_pass = all(
            any(
                cam.get("max_points", 0) > 0
                for cam in object_visibility["objects"].get(key, {}).values()
            )
            for key in ("lego_duplo", "tennis_ball", "open_box")
        )

    contact_chain = (
        behavior_flags["approach"],
        behavior_flags["contact"],
        behavior_flags["grasp"],
        behavior_flags["lift"],
        behavior_flags["place"],
    )
    contact_chain_text = (
        f"approach={contact_chain[0]}, contact={contact_chain[1]}, grasp={contact_chain[2]}, "
        f"lift={contact_chain[3]}, place={contact_chain[4]}"
    )

    if any_success:
        blocker = "No blocker for the success metric in this rollout; success appeared in H5."
        next_gate = (
            "Repeat on multiple seeds and compare against official ManiSkill observations/actions."
        )
    elif task_info_blocker:
        blocker = task_info_blocker
        next_gate = "Add direct H5 behavior fields or fix task_info serialization, then rerun the 400-step gate."
    elif not any(contact_chain):
        blocker = (
            "Behavioral blocker: the 400-step rollout saved artifacts and actions, but decoded "
            "per-step evidence did not show normal-condition approach/contact/grasp/lift/place."
        )
        next_gate = (
            "Compare MolmoSpaces observations/action normalization against official ManiSkill "
            "packets before claiming task behavior progress."
        )
    else:
        blocker = (
            "Behavioral blocker: partial behavior appeared, but the official in-box success metric "
            "did not become true."
        )
        next_gate = "Inspect the first missing behavior stage and align robot/camera/object fidelity before rerun."

    evidence = {
        "API/server": status_text(
            metrics["act_server_probe"].get("get_status_http") == 200 or raw_act_call_count > 0,
            f"server probe/log available; raw /act calls in selected run={raw_act_call_count}",
            f"server probe failed and raw /act calls in selected run={raw_act_call_count}",
        ),
        "sim stepping": status_text(
            ran_400_step_gate,
            f"{num_timepoints} saved timesteps for the 400-step gate",
            f"only {num_timepoints} saved timesteps; expected at least {expected_min_timepoints}",
        ),
        "artifact saved": status_text(
            h5_path.exists() and len(video_paths) == 3 and behavior_log_path.exists(),
            "H5, three camera MP4s, and behavior JSONL are present",
            "one or more required artifacts are missing",
        ),
        "object visibility": status_text(
            visibility_pass,
            "lego, tennis ball, and open box have nonzero image points",
            "one or more task objects lack nonzero image-point evidence",
        ),
        "approach/contact/grasp/lift/place": status_text(
            all(contact_chain),
            contact_chain_text,
            contact_chain_text,
        ),
        "success metric": status_text(
            any_success,
            f"success appeared; last_success={last_success}, reward_sum={reward_sum}",
            f"success never appeared; last_success={last_success}, reward_sum={reward_sum}",
        ),
        "teacher-facing result": status_text(
            any_success and all(contact_chain),
            "successful episode with decoded behavior chain",
            "no teacher-facing successful official-YAM-box episode evidence",
        ),
    }

    metrics.update(
        {
            "traj_key": "traj_0",
            "obs_scene": obs_scene,
            "num_timepoints": num_timepoints,
            "expected_min_timepoints": expected_min_timepoints,
            "ran_400_step_gate": ran_400_step_gate,
            "video_paths": video_paths,
            "side_by_side_video_path": side_by_side_video,
            "raw_action_entries_total": len(raw_entries_all),
            "raw_action_entries_selected": raw_act_call_count,
            "raw_action_segment_path": str(raw_action_segment_path),
            "commanded_action_nonempty_count": action_nonempty,
            "qpos_delta_norm": qpos_delta_norm,
            "qpos_first": qpos_first,
            "qpos_last": qpos_last,
            "success_metric": {
                "summary": summarize_array(success),
                "any_success": any_success,
                "last_success": last_success,
                "reward_sum": reward_sum,
                "last_reward": last_reward,
                "terminated": summarize_array(terminated),
                "truncated": summarize_array(truncated),
                "fail": summarize_array(fail),
            },
            "object_visibility": object_visibility,
            "behavior_summary": behavior_summary,
            "behavior_flags": behavior_flags,
            "first_step": behavior_steps[0] if behavior_steps else None,
            "last_step": behavior_steps[-1] if behavior_steps else None,
            "task_info_blocker": task_info_blocker,
            "blocker": blocker,
            "next_gate": next_gate,
            "evidence": evidence,
        }
    )

    metrics_path.write_text(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True) + "\n")
    write_report(report_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--h5", default=None)
    parser.add_argument("--raw-action-log", default=str(DEFAULT_RAW_ACTION_LOG))
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    metrics = analyze(args)
    print(
        json.dumps(
            {"metrics_path": metrics["metrics_path"], "report_path": metrics["report_path"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/molmospaces-matplotlib")
    os.environ.setdefault("NLTK_DATA", "/tmp/molmospaces-nltk-data")
    main()
