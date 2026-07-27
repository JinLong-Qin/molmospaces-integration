"""Gate 1D-1: dual-object reachability and camera-visibility diagnostic.

This is a diagnostic smoke only. It does not execute a policy, step the task,
collect a demonstration, check collision-free paths, or claim task success.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

if "MOLMOSPACES_NLTK_DATA" in os.environ:
    os.environ.setdefault("NLTK_DATA", os.environ["MOLMOSPACES_NLTK_DATA"])

import nltk

nltk.download = lambda *args, **kwargs: True

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.base_packing_configs import PackingDataGenConfig
from molmo_spaces.configs.camera_configs import AllCameraSystems, BimanualYamCameraSystem
from molmo_spaces.configs.robot_configs import BimanualYamRobotConfig
from molmo_spaces.tasks.packing_task_sampler import PackingTaskSampler
from molmo_spaces.utils.grasps import (
    get_grasp_libraries_for_object,
    get_pickup_grasps,
    has_valid_pickup_grasps,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.object_metadata import ObjectMeta


PICKUP_UIDS = ("Tomato_1", "Potato_1")
CAMERAS = ("exo_camera", "left_wrist_camera", "right_wrist_camera")


class BimanualYamPackingConfig(PackingDataGenConfig):
    camera_config: AllCameraSystems = BimanualYamCameraSystem()


class DualPickupDiagnosticSampler(PackingTaskSampler):
    """Use the existing from-set machinery but activate both staged pickups."""

    @property
    def active_added_pickup_names(self) -> list[str]:
        return list(self._added_pickup_names)


def build_config(seed: int) -> BimanualYamPackingConfig:
    config = BimanualYamPackingConfig(
        scene_dataset="ithor",
        data_split="train",
        robot_config=BimanualYamRobotConfig(),
        camera_config=BimanualYamCameraSystem(),
        num_envs=1,
        num_workers=1,
        seed=seed,
    )
    sampler_cfg = config.task_sampler_config
    sampler_cfg.task_sampler_class = DualPickupDiagnosticSampler
    sampler_cfg.added_pickup_objects = list(PICKUP_UIDS)
    sampler_cfg.num_added_pickups = 2
    sampler_cfg.episodes_per_added_pickup = 100
    sampler_cfg.grasp_libraries = ["droid"]
    sampler_cfg.box_uids = ["Box_24"]
    return config


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def copy_qpos(robot_view) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in robot_view.get_qpos_dict().items()}


def solve_ik(robot, gripper: str, arm: str, target: np.ndarray, q0, base_pose):
    return robot.kinematics.ik(
        gripper,
        target,
        [arm],
        {key: value.copy() for key, value in q0.items()},
        base_pose.copy(),
        max_iter=5000,
        damping=1e-6,
        dt=0.12,
    )


def rank_grasps(robot_view, obj, grasps: np.ndarray, gripper: str, policy_cfg) -> np.ndarray:
    tcp_pose = robot_view.get_move_group(gripper).leaf_frame_to_world.copy()
    tcp_rel = np.linalg.inv(tcp_pose)[None, ...] @ grasps
    pos_cost = np.linalg.norm(tcp_rel[:, :3, 3], axis=1)
    rot_cost = R.from_matrix(tcp_rel[:, :3, :3]).magnitude() * 180.0 / np.pi
    vertical_cost = grasps[:, 2, 2]
    object_rel = np.linalg.inv(obj.pose)[None, ...] @ grasps
    com_cost = np.linalg.norm(object_rel[:, :3, 3], axis=1)
    total = (
        policy_cfg.grasp_pos_cost_weight * pos_cost
        + policy_cfg.grasp_rot_cost_weight * rot_cost
        + policy_cfg.grasp_vertical_cost_weight * vertical_cost
        + policy_cfg.grasp_com_dist_cost_weight * com_cost
    )
    return np.argsort(total)


def solve_object_grasp(
    env,
    obj,
    gripper: str,
    arm: str,
    q0,
    base_pose,
    max_candidates: int,
    pregrasp_offset: float,
) -> dict[str, Any]:
    robot = env.current_robot
    robot.robot_view.set_qpos_dict({key: value.copy() for key, value in q0.items()})
    mujoco.mj_fwdPosition(env.current_model, env.current_data)
    grasps = get_pickup_grasps(
        env,
        obj,
        include_flipped=True,
        grasp_libraries=["droid"],
    )
    order = rank_grasps(robot.robot_view, obj, grasps, gripper, env.current_robot.exp_config.policy_config)
    attempts = 0
    for rank, candidate_index in enumerate(order[:max_candidates]):
        attempts += 1
        grasp = grasps[candidate_index].copy()
        pregrasp = grasp.copy()
        pregrasp[:3, 3] -= pregrasp_offset * pregrasp[:3, 2]
        pre_solution = solve_ik(robot, gripper, arm, pregrasp, q0, base_pose)
        if pre_solution is None:
            continue
        grasp_solution = solve_ik(robot, gripper, arm, grasp, q0, base_pose)
        if grasp_solution is None:
            continue
        return {
            "success": True,
            "total_grasps": len(grasps),
            "attempts": attempts,
            "candidate_rank": rank,
            "candidate_index": int(candidate_index),
            "pregrasp_pose": pregrasp,
            "grasp_pose": grasp,
            "pregrasp_solution": pre_solution,
            "grasp_solution": grasp_solution,
        }
    return {
        "success": False,
        "total_grasps": len(grasps),
        "attempts": attempts,
        "candidate_rank": None,
        "candidate_index": None,
    }


def solve_position_only_pair(
    env,
    gripper: str,
    arm: str,
    first_pose: np.ndarray,
    second_pose: np.ndarray,
    q0,
    base_pose,
) -> dict[str, Any]:
    """Keep target positions but use the arm's q0 TCP orientation as an ablation."""
    robot = env.current_robot
    robot.robot_view.set_qpos_dict({key: value.copy() for key, value in q0.items()})
    mujoco.mj_fwdPosition(env.current_model, env.current_data)
    q0_rotation = robot.robot_view.get_move_group(gripper).leaf_frame_to_world[:3, :3].copy()
    first = first_pose.copy()
    second = second_pose.copy()
    first[:3, :3] = q0_rotation
    second[:3, :3] = q0_rotation
    first_solution = solve_ik(robot, gripper, arm, first, q0, base_pose)
    second_solution = solve_ik(robot, gripper, arm, second, q0, base_pose)
    return {
        "success": first_solution is not None and second_solution is not None,
        "first_pose": first,
        "second_pose": second,
        "first_solution": first_solution,
        "second_solution": second_solution,
    }


def diagnose_object_position_only(
    env,
    obj,
    gripper: str,
    arm: str,
    q0,
    base_pose,
    pregrasp_offset: float,
) -> dict[str, Any]:
    robot = env.current_robot
    robot.robot_view.set_qpos_dict({key: value.copy() for key, value in q0.items()})
    mujoco.mj_fwdPosition(env.current_model, env.current_data)
    grasps = get_pickup_grasps(
        env, obj, include_flipped=True, grasp_libraries=["droid"]
    )
    order = rank_grasps(robot.robot_view, obj, grasps, gripper, env.current_robot.exp_config.policy_config)
    index = int(order[0])
    grasp = grasps[index].copy()
    pregrasp = grasp.copy()
    pregrasp[:3, 3] -= pregrasp_offset * pregrasp[:3, 2]
    result = solve_position_only_pair(
        env, gripper, arm, pregrasp, grasp, q0, base_pose
    )
    result["candidate_index"] = index
    return result


def solve_box_placement(
    env,
    obj,
    box,
    selected_grasp: np.ndarray | None,
    gripper: str,
    arm: str,
    q0,
    base_pose,
    place_offset: float,
) -> dict[str, Any]:
    if selected_grasp is None:
        return {"success": False, "skipped": "no reachable object grasp"}

    model, data = env.current_model, env.current_data
    box_center, box_size = body_aabb(model, data, box.body_id)
    obj_center, obj_size = body_aabb(model, data, obj.body_id)
    box_top_z = float(box_center[2] + box_size[2] / 2.0)
    obj_bottom_z = float(obj_center[2] - obj_size[2] / 2.0)
    clearance = max(float(selected_grasp[2, 3] - obj_bottom_z), 0.0)

    preplace = selected_grasp.copy()
    preplace[:2, 3] = box.position[:2]
    preplace[2, 3] = box_top_z + clearance + place_offset
    preplace[:3, 3] += selected_grasp[:3, 3] - obj.position
    place = preplace.copy()
    place[2, 3] = box_top_z + clearance

    robot = env.current_robot
    pre_solution = solve_ik(robot, gripper, arm, preplace, q0, base_pose)
    place_solution = solve_ik(robot, gripper, arm, place, q0, base_pose)
    return {
        "success": pre_solution is not None and place_solution is not None,
        "box_top_z": box_top_z,
        "object_clearance": clearance,
        "preplace_pose": preplace,
        "place_pose": place,
        "preplace_solution": pre_solution,
        "place_solution": place_solution,
    }


def solve_box_opening_grid(
    env,
    box,
    gripper: str,
    arm: str,
    q0,
    base_pose,
) -> dict[str, Any]:
    """Find an arm-specific reachable point over the shared box opening."""
    robot = env.current_robot
    robot.robot_view.set_qpos_dict({key: value.copy() for key, value in q0.items()})
    mujoco.mj_fwdPosition(env.current_model, env.current_data)
    tcp_pose = robot.robot_view.get_move_group(gripper).leaf_frame_to_world.copy()
    box_center, box_size = body_aabb(env.current_model, env.current_data, box.body_id)
    box_top_z = float(box_center[2] + box_size[2] / 2.0)

    candidates = []
    for x_fraction in (-0.25, 0.0, 0.25):
        for y_fraction in (-0.25, 0.0, 0.25):
            staging_xyz = np.array(
                [
                    box_center[0] + x_fraction * box_size[0],
                    box_center[1] + y_fraction * box_size[1],
                    box_center[2] + 0.04,
                ]
            )
            interior_xyz = np.array(
                [
                    staging_xyz[0],
                    staging_xyz[1],
                    box_center[2],
                ]
            )
            candidates.append(
                (np.linalg.norm(staging_xyz - tcp_pose[:3, 3]), staging_xyz, interior_xyz)
            )
    candidates.sort(key=lambda item: item[0])

    failures = []
    for attempt, (_, staging_xyz, interior_xyz) in enumerate(candidates, start=1):
        approach = tcp_pose.copy()
        interior = tcp_pose.copy()
        approach[:3, 3] = staging_xyz
        interior[:3, 3] = interior_xyz
        result = solve_position_only_pair(
            env, gripper, arm, approach, interior, q0, base_pose
        )
        if result["success"]:
            result.update(
                {
                    "attempts": attempt,
                    "candidate_count": len(candidates),
                    "box_center": box_center,
                    "box_size": box_size,
                    "box_top_z": box_top_z,
                }
            )
            return result
        failures.append(
            {
                "staging_xyz": staging_xyz,
                "interior_xyz": interior_xyz,
                "staging_ok": result["first_solution"] is not None,
                "interior_ok": result["second_solution"] is not None,
            }
        )
    return {
        "success": False,
        "attempts": len(candidates),
        "candidate_count": len(candidates),
        "box_center": box_center,
        "box_size": box_size,
        "box_top_z": box_top_z,
        "failures": failures,
    }


def capture_state(env, label: str, object_names: list[str], out_dir: Path) -> dict[str, Any]:
    env.camera_manager.registry.update_all_cameras(env)
    result: dict[str, Any] = {"label": label, "cameras": {}}
    for camera_name in CAMERAS:
        frame = env.render_rgb_frame(camera_name)
        path = out_dir / f"{label}_{camera_name}.png"
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)
        visibility = env.check_visibility(camera_name, *object_names)
        result["cameras"][camera_name] = {
            "image": path.name,
            "shape": list(frame.shape),
            "pixel_range": int(np.ptp(frame)),
            "nonblank": bool(frame.size > 0 and np.ptp(frame) > 1),
            "visibility": visibility,
        }
    return result


def assign_sides(robot_view, objects_by_uid: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    base_from_world = np.linalg.inv(robot_view.base.pose)
    positions = {}
    for uid, obj in objects_by_uid.items():
        world = np.r_[obj.position, 1.0]
        positions[uid] = (base_from_world @ world)[:3]
    ordered = sorted(positions, key=lambda uid: (-positions[uid][1], uid))
    return {"left": ordered[0], "right": ordered[1]}, positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--house-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-grasp-candidates", type=int, default=32)
    parser.add_argument("--visibility-threshold", type=float, default=1e-4)
    parser.add_argument(
        "--out-dir",
        default="runtime/ithor_bimanual_yam/artifacts/gate_1d1/seed101",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args.seed)
    config.output_dir = str(out_dir)
    sampler = DualPickupDiagnosticSampler(config)
    report: dict[str, Any] = {
        "gate": "1D-1",
        "claim_boundary": (
            "Diagnostic only: static placement, CPU kinematic IK, and segmentation visibility. "
            "No collision-free path, policy rollout, grasp execution, demonstration, or task success."
        ),
        "house_index": args.house_index,
        "seed": args.seed,
        "pickup_uids_requested": list(PICKUP_UIDS),
        "thresholds": {
            "visibility_fraction": args.visibility_threshold,
            "max_grasp_candidates": args.max_grasp_candidates,
        },
    }

    try:
        task = sampler.sample_task(force_advance_scene=True, house_index=args.house_index)
        if task is None:
            raise RuntimeError("sample_task returned None")
        env = sampler.env
        mujoco.mj_forward(env.current_model, env.current_data)
        robot = env.current_robot
        robot_view = robot.robot_view
        object_manager = env.object_managers[env.current_batch_index]

        if len(sampler._added_pickup_names) != 2:
            raise RuntimeError(f"expected 2 added pickups, got {sampler._added_pickup_names}")
        uid_to_name = dict(zip(sampler._added_pickup_uids, sampler._added_pickup_names, strict=True))
        if set(uid_to_name) != set(PICKUP_UIDS):
            raise RuntimeError(f"unexpected pickup UID mapping: {uid_to_name}")
        objects_by_uid = {
            uid: object_manager.get_object_by_name(name) for uid, name in uid_to_name.items()
        }
        box_name = config.task_config.place_receptacle_name
        box = object_manager.get_object_by_name(box_name)

        assignment, base_positions = assign_sides(robot_view, objects_by_uid)
        report["scene"] = {
            "pickup_names": uid_to_name,
            "box_name": box_name,
            "base_frame_positions": base_positions,
            "assignment": assignment,
            "robot_base_pose": robot_view.base.pose.copy(),
        }
        report["assets"] = {}
        for uid in PICKUP_UIDS:
            annotation = ObjectMeta.annotation(uid)
            report["assets"][uid] = {
                "category": annotation.get("category"),
                "primary_property": annotation.get("primaryProperty"),
                "bounding_box": annotation.get("boundingBox"),
                "grasp_libraries": get_grasp_libraries_for_object(uid),
                "valid_droid_grasp": has_valid_pickup_grasps(
                    uid, num_grasps=1, grasp_libraries=("droid",)
                ),
            }

        object_names = [uid_to_name[uid] for uid in PICKUP_UIDS] + [box_name]
        report["visibility"] = {
            "initial": capture_state(env, "initial", object_names, out_dir)
        }

        q0 = copy_qpos(robot_view)
        base_pose = robot_view.base.pose.copy()
        pregrasp_offset = float(config.policy_config.pregrasp_z_offset)
        place_offset = float(config.policy_config.place_z_offset)
        arm_specs = {
            "left": ("left_gripper", "left_arm"),
            "right": ("right_gripper", "right_arm"),
        }
        report["ik"] = {"objects": {}, "box": {}}
        report["diagnostics"] = {"droid_orientation": {"objects": {}, "box": {}}}
        for side, (gripper, arm) in arm_specs.items():
            report["ik"]["objects"][side] = {}
            report["diagnostics"]["droid_orientation"]["objects"][side] = {}
            for relation, uid in (
                ("own", assignment[side]),
                ("other", assignment["right" if side == "left" else "left"]),
            ):
                droid_result = solve_object_grasp(
                    env,
                    objects_by_uid[uid],
                    gripper,
                    arm,
                    q0,
                    base_pose,
                    args.max_grasp_candidates,
                    pregrasp_offset,
                )
                droid_result["uid"] = uid
                report["diagnostics"]["droid_orientation"]["objects"][side][relation] = (
                    droid_result
                )
                report["ik"]["objects"][side][relation] = diagnose_object_position_only(
                    env,
                    objects_by_uid[uid],
                    gripper,
                    arm,
                    q0,
                    base_pose,
                    pregrasp_offset,
                )
                report["ik"]["objects"][side][relation]["uid"] = uid

            own_droid = report["diagnostics"]["droid_orientation"]["objects"][side]["own"]
            selected_grasp = own_droid.get("grasp_pose") if own_droid["success"] else None
            own_obj = objects_by_uid[assignment[side]]
            report["diagnostics"]["droid_orientation"]["box"][side] = solve_box_placement(
                env,
                own_obj,
                box,
                selected_grasp,
                gripper,
                arm,
                q0,
                base_pose,
                place_offset,
            )
            report["diagnostics"]["droid_orientation"]["box"][side]["uid"] = assignment[
                side
            ]
            report["ik"]["box"][side] = solve_box_opening_grid(
                env, box, gripper, arm, q0, base_pose
            )
            report["ik"]["box"][side]["uid"] = assignment[side]

        own_left = report["ik"]["objects"]["left"]["own"]
        own_right = report["ik"]["objects"]["right"]["own"]
        dual_reach_available = own_left["success"] and own_right["success"]
        if dual_reach_available:
            robot_view.set_qpos_dict({key: value.copy() for key, value in q0.items()})
            robot_view.set_qpos_dict(
                {
                    "left_arm": own_left["first_solution"]["left_arm"],
                    "right_arm": own_right["first_solution"]["right_arm"],
                }
            )
            mujoco.mj_forward(env.current_model, env.current_data)
            report["visibility"]["dual_reach"] = capture_state(
                env, "dual_reach", object_names, out_dir
            )
        else:
            report["visibility"]["dual_reach"] = {
                "skipped": "one or both own-object pregrasp IK checks failed"
            }

        robot_view.set_qpos_dict({key: value.copy() for key, value in q0.items()})
        mujoco.mj_forward(env.current_model, env.current_data)
        env.camera_manager.registry.update_all_cameras(env)

        asset_pass = all(report["assets"][uid]["valid_droid_grasp"] for uid in PICKUP_UIDS)
        ik_pass = all(
            report["ik"]["objects"][side]["own"]["success"]
            and report["ik"]["box"][side]["success"]
            for side in ("left", "right")
        )
        threshold = args.visibility_threshold
        initial = report["visibility"]["initial"]["cameras"]
        initial_top_pass = all(
            initial["exo_camera"]["visibility"].get(name, 0.0) >= threshold
            for name in object_names
        )
        all_images_nonblank = all(item["nonblank"] for item in initial.values())
        camera_pass = False
        if dual_reach_available:
            dual = report["visibility"]["dual_reach"]["cameras"]
            all_images_nonblank = all_images_nonblank and all(
                item["nonblank"] for item in dual.values()
            )
            left_name = uid_to_name[assignment["left"]]
            right_name = uid_to_name[assignment["right"]]
            camera_pass = (
                all_images_nonblank
                and initial_top_pass
                and dual["left_wrist_camera"]["visibility"].get(left_name, 0.0) >= threshold
                and dual["right_wrist_camera"]["visibility"].get(right_name, 0.0) >= threshold
                and all(
                    dual["exo_camera"]["visibility"].get(name, 0.0) >= threshold
                    for name in object_names
                )
            )

        report["gate_result"] = {
            "asset_pass": asset_pass,
            "ik_pass": ik_pass,
            "camera_pass": camera_pass,
            "strict_pass": asset_pass and ik_pass and camera_pass,
        }
    except Exception as exc:
        report["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["gate_result"] = {
            "asset_pass": False,
            "ik_pass": False,
            "camera_pass": False,
            "strict_pass": False,
        }
        raise
    finally:
        report_path = out_dir / "report.json"
        report_path.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
        sampler.close()
        print(json.dumps(report.get("gate_result", {}), indent=2))
        print(f"report={report_path}")


if __name__ == "__main__":
    main()
