"""Standalone source-aligned MolmoAct2 YAM box rollout analysis.

This avoids importing the heavier MolmoSpaces config stack after rollout and
only reads the fresh H5/video/raw-action artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ARTIFACT_DIR = Path("artifacts/molmospaces/molmoact2_yam_integration")
REPORT_PATH = ARTIFACT_DIR / "final_source_aligned_video_report_20260623.md"
RAW_ACTION_LOG = Path("artifacts/molmospaces/molmoact2_official_yam_box_debug/raw_actions.jsonl")

OFFICIAL_TASK_SOURCE = (
    "/home/c/project/paper_reproductions/official/artifacts/molmospaces/"
    "molmoact2_yam_integration_analysis_20260620_142443/molmoact2/"
    "sim_eval/tasks/yam_tasks/bimanual_put_everything_in_box.py"
)
OFFICIAL_ROBOT_SOURCE = (
    "/home/c/project/paper_reproductions/official/artifacts/molmospaces/"
    "molmoact2_yam_integration_analysis_20260620_142443/molmoact2/"
    "sim_eval/robots/bimanual_yam.py"
)
OBJECT_NAMES = ("obj_073-a_lego_duplo", "obj_056_tennis_ball")
OBJECT_ALIASES = {
    "lego_duplo": "obj_073-a_lego_duplo",
    "tennis_ball": "obj_056_tennis_ball",
    "open_box": "open_box",
}
CAMERAS = ("top_cam", "left_cam", "right_cam")
APPROACH_DELTA_M = 0.02
NEAR_CONTACT_M = 0.06
LIFT_DELTA_M = 0.03


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(val) for val in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def decode_json_row(row: Any) -> dict[str, Any]:
    raw = bytes(np.asarray(row, dtype=np.uint8)).split(b"\x00", 1)[0].decode("utf-8")
    return json.loads(raw) if raw else {}


def load_raw_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def current_raw_segment(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_start = None
    for idx, entry in enumerate(entries):
        if int(entry.get("call_index", -1)) == 0:
            last_start = idx
    return entries[last_start:] if last_start is not None else entries


def collect_video_paths(run_dir: Path) -> dict[str, str]:
    house_dir = run_dir / "house_0"
    paths: dict[str, str] = {}
    for camera in CAMERAS:
        matches = sorted(house_dir.glob(f"episode_*_{camera}_batch_*.mp4"))
        if matches:
            paths[camera] = str(matches[0])
    return paths


def compose_three_camera_video(video_paths: dict[str, str], output_path: Path) -> str | None:
    if len(video_paths) != 3 or shutil.which("ffmpeg") is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_paths["top_cam"],
        "-i",
        video_paths["left_cam"],
        "-i",
        video_paths["right_cam"],
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
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return str(output_path)


def summarize_visibility(traj: h5py.Group) -> dict[str, Any]:
    root_path = "obs/extra/object_image_points"
    if root_path not in traj:
        return {"available": False}
    root = traj[root_path]
    summary: dict[str, Any] = {"available": True, "objects": {}}
    for alias in OBJECT_ALIASES:
        if alias not in root:
            continue
        summary["objects"][alias] = {}
        for camera in CAMERAS:
            path = f"{alias}/{camera}/num_points"
            if path in root:
                values = np.asarray(root[path]).reshape(-1)
                summary["objects"][alias][camera] = {
                    "max_points": int(values.max()) if values.size else 0,
                    "visible_steps": int(np.count_nonzero(values > 0)),
                }
    return summary


def object_position(info: dict[str, Any], object_name: str) -> list[float] | None:
    behavior_pose = (
        info.get("official_yam_behavior", {})
        .get("object_poses", {})
        .get(object_name, {})
        .get("position")
    )
    if behavior_pose is not None:
        return behavior_pose
    return (
        info.get("official_yam_box", {}).get("per_object", {}).get(object_name, {}).get("position")
    )


def analyze_task_infos(
    task_infos: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first_positions = (
        {name: object_position(task_infos[0], name) for name in OBJECT_NAMES} if task_infos else {}
    )
    first_distances: dict[str, dict[str, float]] = {}
    min_distances = {
        name: {"left_gripper": math.inf, "right_gripper": math.inf} for name in OBJECT_NAMES
    }
    max_lift_delta = {name: 0.0 for name in OBJECT_NAMES}
    event_steps = {
        "approach_delta": [],
        "near_contact_distance": [],
        "contact": [],
        "grasp": [],
        "lift": [],
        "place": [],
    }
    rows: list[dict[str, Any]] = []

    for idx, info in enumerate(task_infos):
        behavior = info.get("official_yam_behavior", {})
        distances = behavior.get("gripper_object_distances", {})
        contacts = behavior.get("contacts", {})
        per_object_metrics = info.get("official_yam_box", {}).get("per_object", {})

        if idx == 0:
            for gripper_name, per_object in distances.items():
                for object_name, distance in per_object.items():
                    first_distances.setdefault(object_name, {})[gripper_name] = float(distance)

        object_positions = {
            name: object_position(info, name) for name in (*OBJECT_NAMES, "open_box")
        }
        for gripper_name, per_object in distances.items():
            for object_name, distance in per_object.items():
                if object_name not in min_distances:
                    continue
                value = float(distance)
                min_distances[object_name][gripper_name] = min(
                    min_distances[object_name][gripper_name], value
                )
                first = first_distances.get(object_name, {}).get(gripper_name)
                if first is not None and first - value >= APPROACH_DELTA_M:
                    event_steps["approach_delta"].append(idx)
                if value <= NEAR_CONTACT_M:
                    event_steps["near_contact_distance"].append(idx)

        for object_name in OBJECT_NAMES:
            start_pos = first_positions.get(object_name)
            current_pos = object_positions.get(object_name)
            if start_pos is not None and current_pos is not None:
                dz = float(current_pos[2] - start_pos[2])
                max_lift_delta[object_name] = max(max_lift_delta[object_name], dz)
                if dz >= LIFT_DELTA_M:
                    event_steps["lift"].append(idx)

            if per_object_metrics.get(object_name, {}).get("in_box", False):
                event_steps["place"].append(idx)

            object_contacts = contacts.get(object_name, {})
            if object_contacts.get("any_gripper_touching", False):
                event_steps["contact"].append(idx)
            for grip_state in object_contacts.get("per_gripper", {}).values():
                if grip_state.get("held", False):
                    event_steps["grasp"].append(idx)

        rows.append(
            {
                "timestep": idx,
                "success": bool(info.get("success", False)),
                "n_in_box": int(info.get("n_in_box", 0)),
                "n_total": int(info.get("n_total", len(OBJECT_NAMES))),
                "object_positions": object_positions,
                "gripper_object_distances": distances,
                "contacts": contacts,
                "official_yam_box": info.get("official_yam_box", {}),
            }
        )

    approach_delta_by_object = {}
    for object_name in OBJECT_NAMES:
        approach_delta_by_object[object_name] = {}
        for gripper_name in ("left_gripper", "right_gripper"):
            first = first_distances.get(object_name, {}).get(gripper_name)
            minimum = min_distances[object_name][gripper_name]
            approach_delta_by_object[object_name][gripper_name] = (
                None if first is None or math.isinf(minimum) else first - minimum
            )

    flags = {
        "approach_delta": bool(event_steps["approach_delta"]),
        "near_contact_distance": bool(event_steps["near_contact_distance"]),
        "contact": bool(event_steps["contact"]),
        "grasp": bool(event_steps["grasp"]),
        "lift": bool(event_steps["lift"]),
        "place": bool(event_steps["place"]),
    }
    summary = {
        "flags": flags,
        "event_steps": {key: sorted(set(value))[:25] for key, value in event_steps.items()},
        "thresholds": {
            "approach_delta_m": APPROACH_DELTA_M,
            "near_contact_distance_m": NEAR_CONTACT_M,
            "lift_delta_m": LIFT_DELTA_M,
        },
        "min_gripper_object_distance_m": {
            object_name: {
                gripper_name: None if math.isinf(distance) else distance
                for gripper_name, distance in per_gripper.items()
            }
            for object_name, per_gripper in min_distances.items()
        },
        "approach_delta_by_object_m": approach_delta_by_object,
        "max_object_lift_delta_m": max_lift_delta,
        "first_positions": first_positions,
        "last_positions": {
            name: object_position(task_infos[-1], name) for name in (*OBJECT_NAMES, "open_box")
        }
        if task_infos
        else {},
    }
    return rows, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def evidence_line(passed: bool, pass_text: str, fail_text: str) -> str:
    return f"PASS: {pass_text}" if passed else f"FAIL: {fail_text}"


def write_report(metrics: dict[str, Any]) -> None:
    evidence = metrics["evidence"]
    flags = metrics["behavior_summary"]["flags"]
    success = metrics["success_metric"]
    videos = metrics["video_paths"]
    report = f"""# MolmoAct2 Source-Aligned YAM Box Video Report

Date: 2026-06-23

## Official Source Facts Used
- Task source: `{OFFICIAL_TASK_SOURCE}`
- Robot source: `{OFFICIAL_ROBOT_SOURCE}`
- Env ID: `BimanualYAMPutEverythingInBox-v1`
- Instruction: `put everything into the box`
- Objects: `obj_073-a_lego_duplo`, `obj_056_tennis_ball`, and static `open_box`
- Source XY setup: lego `(-0.30, +0.22)`, tennis ball `(-0.30, -0.22)`, box `(-0.15, 0.0)`, spawn noise `0.02`
- Source success: both object centers inside the box interior in X/Y and within the box Z band
- Robot reset/home: base pose `[-0.65, 0, 0.01]`, home joint qpos zeros
- Camera triad: `top_cam`, `left_cam`, `right_cam`, 640x360; top HFOV 69.4 deg, wrist HFOV 87.0 deg
- Approximation: MolmoSpaces uses MuJoCo primitives and the compact table scene in `examples/molmoact2_official_yam_box/scene.xml`; the executable Z/table geometry is a MuJoCo approximation, while the source task XY/object/camera semantics are preserved.

## Exact Changed Files
- `molmo_spaces/data_generation/config/molmoact2_official_yam_box_config.py`
- `scripts/smoke_molmoact2_official_yam_box.py`
- `scripts/analyze_molmoact2_source_aligned_yam_box_rollout.py`

Unchanged after inspection: `molmo_spaces/configs/camera_configs.py`, `molmo_spaces/policy/learned_policy/molmoact2_yam_policy.py`, `scripts/datagen/run_pipeline.py`, `examples/molmoact2_official_yam_box/scene.xml`, `examples/molmoact2_official_yam_box/scene_metadata.json`.

## Commands Run
- Smoke/source contract: `PYTHONPATH=. .venv/bin/python scripts/smoke_molmoact2_official_yam_box.py`
- Compile check: `PYTHONPATH=. .venv/bin/python -m py_compile molmo_spaces/data_generation/config/molmoact2_official_yam_box_config.py molmo_spaces/configs/camera_configs.py molmo_spaces/policy/learned_policy/molmoact2_yam_policy.py scripts/datagen/run_pipeline.py scripts/smoke_molmoact2_official_yam_box.py scripts/analyze_molmoact2_official_yam_box_rollout.py scripts/analyze_molmoact2_source_aligned_yam_box_rollout.py`
- Server probe: `.venv/bin/python - <<'PY' ... requests.get('http://127.0.0.1:8202/act') ... PY`
- Rollout: `PYTHONPATH=. .venv/bin/python scripts/datagen/run_pipeline.py --config MolmoAct2OfficialYamBoxDataGenConfig --robot bimanual_yam --policy molmoact2_yam --task_horizon 400 --run_name_prefix molmoact2_official_yam_source_aligned_20260623`
- Analysis: `PYTHONPATH=. .venv/bin/python scripts/analyze_molmoact2_source_aligned_yam_box_rollout.py --run-dir {metrics["run_dir"]}`

## Artifact Paths
- Run dir: `{metrics["run_dir"]}`
- H5: `{metrics["h5_path"]}`
- top_cam MP4: `{videos.get("top_cam")}`
- left_cam MP4: `{videos.get("left_cam")}`
- right_cam MP4: `{videos.get("right_cam")}`
- Combined three-camera MP4: `{metrics.get("combined_video_path")}`
- Metrics JSON: `{metrics["metrics_path"]}`
- Behavior JSONL: `{metrics["behavior_log_path"]}`
- Selected raw `/act` JSONL: `{metrics["raw_action_segment_path"]}`
- Rollout log: `artifacts/molmospaces/molmoact2_yam_integration/molmoact2_official_yam_source_aligned_20260623_rollout.log`

## Evidence Matrix
| Layer | Result |
|---|---|
| API/server | {evidence["API/server"]} |
| sim stepping | {evidence["sim stepping"]} |
| artifact generation | {evidence["artifact generation"]} |
| task-directed behavior | {evidence["task-directed behavior"]} |
| official success metric | {evidence["official success metric"]} |

## Behavior And Outcome
- Saved timesteps: {metrics["num_timepoints"]}
- Raw `/act` calls selected for this run: {metrics["raw_act_calls_selected"]}
- Approach by distance reduction >= {APPROACH_DELTA_M} m: {flags["approach_delta"]}
- Near-contact distance <= {NEAR_CONTACT_M} m: {flags["near_contact_distance"]}
- Gripper contact: {flags["contact"]}
- Grasp/held: {flags["grasp"]}
- Lift >= {LIFT_DELTA_M} m: {flags["lift"]}
- Place/in-box: {flags["place"]}
- Official success any/last: {success["any_success"]} / {success["last_success"]}
- Official reward max/last/sum: {success["max_reward"]} / {success["last_reward"]} / {success["reward_sum"]}

Task outcome: failed. The corrected route generated a fresh video and 400-step H5 against the live MolmoAct2 YAM `/act` server, but neither object entered the box and the official success metric stayed false for every saved timestep.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    h5_path = (
        Path(args.h5).resolve() if args.h5 else run_dir / "house_0/trajectories_batch_1_of_1.h5"
    )
    ts = args.timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        ARTIFACT_DIR / f"molmoact2_official_yam_source_aligned_20260623_{ts}_metrics.json"
    )
    behavior_log_path = (
        ARTIFACT_DIR / f"molmoact2_official_yam_source_aligned_20260623_{ts}_behavior_log.jsonl"
    )
    raw_segment_path = (
        ARTIFACT_DIR
        / f"molmoact2_official_yam_source_aligned_20260623_{ts}_raw_actions_selected.jsonl"
    )
    combined_video_path = (
        ARTIFACT_DIR / f"molmoact2_official_yam_source_aligned_20260623_{ts}_three_camera.mp4"
    )

    raw_entries_all = load_raw_entries(Path(args.raw_action_log))
    raw_segment = current_raw_segment(raw_entries_all)
    write_jsonl(raw_segment_path, raw_segment)

    video_paths = collect_video_paths(run_dir)
    combined_video = compose_three_camera_video(video_paths, combined_video_path)

    with h5py.File(h5_path, "r") as h5_file:
        traj = h5_file[sorted(h5_file.keys())[0]]
        rewards = np.asarray(traj["rewards"], dtype=float)
        success = np.asarray(traj["success"], dtype=bool)
        task_infos = [decode_json_row(row) for row in traj["obs/extra/task_info"]]
        visibility = summarize_visibility(traj)

    behavior_rows, behavior_summary = analyze_task_infos(task_infos)
    for idx, row in enumerate(behavior_rows):
        if idx < len(rewards):
            row["reward"] = float(rewards[idx])
        if idx < len(success):
            row["success"] = bool(success[idx])
    write_jsonl(behavior_log_path, behavior_rows)

    num_timepoints = int(len(success))
    any_success = bool(np.any(success))
    last_success = bool(success[-1]) if success.size else False
    reward_sum = float(rewards.sum()) if rewards.size else 0.0
    max_reward = float(rewards.max()) if rewards.size else 0.0
    last_reward = float(rewards[-1]) if rewards.size else 0.0
    artifact_ok = h5_path.exists() and len(video_paths) == 3 and combined_video is not None
    sim_ok = num_timepoints >= 401
    behavior_flags = behavior_summary["flags"]
    behavior_text = (
        f"approach_delta={behavior_flags['approach_delta']}, "
        f"near_contact={behavior_flags['near_contact_distance']}, "
        f"contact={behavior_flags['contact']}, grasp={behavior_flags['grasp']}, "
        f"lift={behavior_flags['lift']}, place={behavior_flags['place']}"
    )

    metrics = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "h5_path": str(h5_path),
        "video_paths": video_paths,
        "combined_video_path": combined_video,
        "metrics_path": str(metrics_path),
        "behavior_log_path": str(behavior_log_path),
        "raw_action_log_path": str(args.raw_action_log),
        "raw_action_segment_path": str(raw_segment_path),
        "raw_act_calls_selected": len(raw_segment),
        "num_timepoints": num_timepoints,
        "visibility": visibility,
        "behavior_summary": behavior_summary,
        "success_metric": {
            "any_success": any_success,
            "last_success": last_success,
            "reward_sum": reward_sum,
            "max_reward": max_reward,
            "last_reward": last_reward,
            "success_count": int(np.count_nonzero(success)),
        },
        "official_sources": {
            "task": OFFICIAL_TASK_SOURCE,
            "robot": OFFICIAL_ROBOT_SOURCE,
        },
    }
    metrics["evidence"] = {
        "API/server": evidence_line(
            len(raw_segment) > 0,
            f"fresh raw /act segment has {len(raw_segment)} calls",
            "no raw /act calls selected",
        ),
        "sim stepping": evidence_line(
            sim_ok,
            f"{num_timepoints} saved timesteps for a 400-step horizon",
            f"{num_timepoints} saved timesteps; expected 401 including reset observation",
        ),
        "artifact generation": evidence_line(
            artifact_ok,
            "H5, top/left/right MP4s, and combined three-camera MP4 are present",
            "one or more H5/video artifacts are missing",
        ),
        "task-directed behavior": evidence_line(
            all(
                behavior_flags[key]
                for key in ("near_contact_distance", "contact", "grasp", "lift", "place")
            ),
            behavior_text,
            behavior_text,
        ),
        "official success metric": evidence_line(
            any_success,
            f"success appeared; last_success={last_success}, max_reward={max_reward}",
            f"success never appeared; last_success={last_success}, max_reward={max_reward}",
        ),
    }

    metrics_path.write_text(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True) + "\n")
    write_report(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--h5", default=None)
    parser.add_argument("--raw-action-log", default=str(RAW_ACTION_LOG))
    parser.add_argument("--timestamp", default="20260623_185706")
    args = parser.parse_args()
    metrics = analyze(args)
    print(
        json.dumps(
            {"metrics_path": metrics["metrics_path"], "report_path": str(REPORT_PATH)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
