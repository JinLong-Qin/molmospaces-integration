"""Gate 1D-3: generate one strict bimanual scripted-expert source demo.

This is an additive experiment under runtime/ithor_bimanual_yam.  It uses
MolmoSpaces' official iTHOR FloorPlan1/Packing placement flow, YAM controllers,
MuJoCo dynamics, sensor/history path, HDF5 writer, and camera-video writer.  The
only task-specific additions are: two gripper-sized THOR pickup assets, explicit
left/right workspace placement, a sequential dual-arm oracle, and strict 2/2
success/replay metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import mujoco
import numpy as np

import check_dual_object_reachability as reach_mod
import validate_tabletop_initialization as init_mod
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.pose import pose_mat_to_7d
from molmo_spaces.utils.save_utils import prepare_episode_for_saving, save_trajectories

PICKUP_UIDS = ("Candle_1", "Apple_29")
ARMS = {
    "left": ("left_arm", "left_gripper"),
    "right": ("right_arm", "right_gripper"),
}
OPEN = 0.041
CLOSED = 0.0
CONTROL_STRIDE = 20
GRIPPER_MAX_INNER_WIDTH_M = 0.082
GRASP_WIDTH_MARGIN_M = 0.010


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def copy_qpos(robot_view):
    return {
        k: np.asarray(v, dtype=np.float64).copy() for k, v in robot_view.get_qpos_dict().items()
    }


def hold_action(robot_view):
    action = copy_qpos(robot_view)
    # YAM exposes two finger joint positions per gripper, but each gripper is
    # driven by one actuator command.  Normalize both sides on every action,
    # including the inactive arm's hold command.
    for gripper in ("left_gripper", "right_gripper"):
        action[gripper] = np.array([float(np.mean(action[gripper]))], dtype=np.float64)
    return action


def solve_position(
    robot, robot_view, side: str, position: np.ndarray, qseed, reference: np.ndarray
):
    arm, gripper = ARMS[side]
    target = reference.copy()
    target[:3, 3] = np.asarray(position, dtype=np.float64)
    solution = robot.kinematics.ik(
        gripper,
        target,
        [arm],
        {k: np.asarray(v).copy() for k, v in qseed.items()},
        robot_view.base.pose.copy(),
        max_iter=5000,
        damping=1e-6,
        dt=0.12,
    )
    if solution is None:
        raise RuntimeError(f"IK failed: side={side}, position={position.tolist()}")
    return np.asarray(solution[arm], dtype=np.float64).copy()


def graspable_width_record(model, data, body_id: int) -> dict[str, Any]:
    _, size = body_aabb(model, data, body_id)
    size = np.asarray(size, dtype=np.float64)
    horizontal = np.sort(size[:2])
    effective_width = float(horizontal[0])
    limit = GRIPPER_MAX_INNER_WIDTH_M - GRASP_WIDTH_MARGIN_M
    return {
        "world_aabb_size_m": size,
        "effective_horizontal_width_m": effective_width,
        "gripper_max_inner_width_m": GRIPPER_MAX_INNER_WIDTH_M,
        "required_margin_m": GRASP_WIDTH_MARGIN_M,
        "acceptance_limit_m": limit,
        "pass": bool(effective_width <= limit),
    }


def place_for_arm_assignment(task, sampler) -> dict[str, Any]:
    """Deterministically place small pickups in separate base-frame half-spaces."""
    env = task.env
    data, model = env.current_data, env.current_model
    rv = env.current_robot.robot_view
    om = env.object_managers[env.current_batch_index]
    uid_to_name = dict(zip(sampler._added_pickup_uids, sampler._added_pickup_names, strict=True))
    if set(uid_to_name) != set(PICKUP_UIDS):
        raise RuntimeError(f"unexpected pickup mapping: {uid_to_name}")
    box_name = sampler.config.task_config.place_receptacle_name
    base = rv.base.pose.copy()
    forward, left = base[:3, 0], base[:3, 1]
    targets_base = {
        "left": np.array([0.42, +0.24, 0.0]),
        "right": np.array([0.42, -0.24, 0.0]),
        # Keep the box at the original dynamically validated center. Moving it
        # changed scene settling and destroyed deterministic grasp replay.
        "box": np.array([0.62, 0.0, 0.0]),
    }
    assignment = {"left": PICKUP_UIDS[0], "right": PICKUP_UIDS[1]}
    before = {}
    after = {}
    width_gate = {}
    for side, uid in assignment.items():
        # ObjectManager returns MlSpacesObject, whose world-space position is
        # intentionally read-only.  Resolve its root free joint for mutation.
        obj = create_mlspaces_body(data, uid_to_name[uid])
        before[uid] = obj.position.copy()
        current_center, current_size = body_aabb(model, data, obj.body_id)
        current_bottom = float(current_center[2] - current_size[2] / 2)
        support_top = float(sampler.accepted_initialization["support_top_z"])
        world_xy = base[:3, 3] + targets_base[side][0] * forward + targets_base[side][1] * left
        # Keep only a sub-millimetre clearance before the deterministic settle;
        # the previous 4 mm drop let round pickup objects roll centimetres while
        # the arm was approaching a grasp computed from the stale pre-drop pose.
        dz = support_top - current_bottom + 0.0005
        obj.position = np.array([world_xy[0], world_xy[1], obj.position[2] + dz])
    box = create_mlspaces_body(data, box_name)
    before["box"] = box.position.copy()
    box_center, box_size = body_aabb(model, data, box.body_id)
    box_bottom = float(box_center[2] - box_size[2] / 2)
    support_top = float(sampler.accepted_initialization["support_top_z"])
    box_world = base[:3, 3] + targets_base["box"][0] * forward + targets_base["box"][1] * left
    box.position = np.array(
        [box_world[0], box_world[1], box.position[2] + support_top - box_bottom + 0.0005]
    )
    mujoco.mj_forward(model, data)
    for side, uid in assignment.items():
        obj = create_mlspaces_body(data, uid_to_name[uid])
        rel = np.linalg.solve(base, obj.pose)[:3, 3]
        after[uid] = {"world_position": obj.position.copy(), "base_position": rel, "side": side}
        width_gate[uid] = graspable_width_record(model, data, obj.body_id)
    box_rel = np.linalg.solve(base, box.pose)[:3, 3]
    after["box"] = {"world_position": box.position.copy(), "base_position": box_rel}
    side_pass = bool(
        after[PICKUP_UIDS[0]]["base_position"][1] > 0.08
        and after[PICKUP_UIDS[1]]["base_position"][1] < -0.08
    )
    width_pass = all(item["pass"] for item in width_gate.values())
    if not side_pass or not width_pass:
        raise RuntimeError(
            f"workspace/graspability gate failed: side={side_pass}, width={width_gate}"
        )
    return {
        "uid_to_name": uid_to_name,
        "box_name": box_name,
        "assignment": assignment,
        "before": before,
        "after": after,
        "width_gate": width_gate,
        "side_assignment_pass": side_pass,
        "graspability_pass": width_pass,
    }


def object_supported_by_box(task, object_name: str, box_name: str) -> bool:
    om = task.env.object_managers[task.env.current_batch_index]
    obj = om.get_object_by_name(object_name)
    box = om.get_object_by_name(box_name)
    on = om.objects_on_receptacle([obj], box.geom_ids)
    return object_name in {item.name for item in on}


def strict_success(task, object_names: list[str], box_name: str) -> dict[str, Any]:
    env, data = task.env, task.env.current_data
    rv = env.current_robot.robot_view
    robot_root = rv.base.root_body_id
    per_object = {}
    for name in object_names:
        obj = create_mlspaces_body(data, name)
        robot_contact = False
        robot_contact_pairs = []
        for contact in data.contact:
            root1 = data.model.body_rootid[data.model.geom_bodyid[contact.geom1]]
            root2 = data.model.body_rootid[data.model.geom_bodyid[contact.geom2]]
            if (root1 == obj.body_id) ^ (root2 == obj.body_id):
                other = root2 if root1 == obj.body_id else root1
                if other == robot_root:
                    robot_contact = True
                    body1 = int(data.model.geom_bodyid[contact.geom1])
                    body2 = int(data.model.geom_bodyid[contact.geom2])
                    robot_contact_pairs.append(
                        {
                            "geom1_id": int(contact.geom1),
                            "geom1_name": data.model.geom(int(contact.geom1)).name,
                            "body1_name": data.model.body(body1).name,
                            "geom2_id": int(contact.geom2),
                            "geom2_name": data.model.geom(int(contact.geom2)).name,
                            "body2_name": data.model.body(body2).name,
                        }
                    )
        supported = object_supported_by_box(task, name, box_name)
        per_object[name] = {
            "supported_by_box": bool(supported),
            "released_from_robot": not robot_contact,
            "robot_contact_pairs": robot_contact_pairs,
            "left_finger_width_m": float(rv.get_move_group("left_gripper").inter_finger_dist),
            "right_finger_width_m": float(rv.get_move_group("right_gripper").inter_finger_dist),
            "pass": bool(supported and not robot_contact),
            "position": obj.position.copy(),
        }
    return {
        "per_object": per_object,
        "count": sum(v["pass"] for v in per_object.values()),
        "total": len(per_object),
        "success": all(v["pass"] for v in per_object.values()),
    }


class Recorder:
    def __init__(self, task, object_names: list[str], box_name: str, record_enabled: bool = True):
        self.task = task
        self.env = task.env
        self.robot = self.env.current_robot
        self.rv = self.robot.robot_view
        self.object_names = object_names
        self.box_name = box_name
        self.commands: list[dict[str, np.ndarray]] = []
        self.records: list[dict[str, Any]] = []
        self.history_states: list[np.ndarray] = []
        self.history_actions: list[dict[str, np.ndarray]] = []
        self.sim_step = 0
        self.record_enabled = record_enabled
        self.probe_trace_enabled = False
        self.probe_trace: list[tuple[dict[str, np.ndarray], str, str]] = []

    def _record(self, action, phase: str, active_side: str):
        # Online sensor/history extraction is intentionally forbidden here.
        # Some sensors update renderer/camera/task-side state and changed the
        # contact-sensitive trajectory relative to the no-sensor dynamic probe.
        # Save the exact successful physics state now; observations are
        # materialized from these snapshots only after strict 2/2 succeeds.
        data = self.env.current_data
        self.history_states.append(snapshot_physics(self.env))
        self.history_actions.append({k: np.asarray(v).copy() for k, v in action.items()})
        self.records.append(
            {
                "sim_step": self.sim_step,
                "phase": phase,
                "active_side": active_side,
                "left_tcp": pose_mat_to_7d(
                    self.rv.get_move_group("left_gripper").leaf_frame_to_world
                ),
                "right_tcp": pose_mat_to_7d(
                    self.rv.get_move_group("right_gripper").leaf_frame_to_world
                ),
                "object_poses": {
                    name: np.concatenate(
                        [
                            create_mlspaces_body(data, name).position,
                            create_mlspaces_body(data, name).quat,
                        ]
                    )
                    for name in self.object_names
                },
            }
        )

    def materialize_history(self):
        """Build task sensor history offline from the successful state trace."""
        final_state = snapshot_physics(self.env)
        initialize_episode_caches(self.task)
        for state, action in zip(self.history_states, self.history_actions, strict=True):
            restore_physics(self.env, state)
            self.task.last_action = {k: np.asarray(v).copy() for k, v in action.items()}
            self.task.action_cache.append(self.task.last_action)
            self.task.episode_step_count += 1
            self.task.get_and_cache_all_step_information()
        restore_physics(self.env, final_state)

    def execute(self, action, phase: str, active_side: str):
        if self.probe_trace_enabled:
            self.probe_trace.append(
                ({k: np.asarray(v).copy() for k, v in action.items()}, phase, active_side)
            )
        self.robot.update_control(action)
        self.robot.compute_control()
        self.env.step(1)
        self.sim_step += 1
        if self.record_enabled:
            self.commands.append({k: np.asarray(v).copy() for k, v in action.items()})
            if self.sim_step % CONTROL_STRIDE == 0:
                self._record(action, phase, active_side)

    def hold(self, side: str, target_q: np.ndarray, gripper_command: float, steps: int, phase: str):
        arm, gripper = ARMS[side]
        for _ in range(steps):
            action = hold_action(self.rv)
            action[arm] = np.asarray(target_q).copy()
            action[gripper] = np.array([gripper_command], dtype=np.float64)
            self.execute(action, phase, side)

    def ramp(
        self,
        side: str,
        start_q: np.ndarray,
        end_q: np.ndarray,
        start_gripper: float,
        end_gripper: float,
        steps: int,
        phase: str,
    ):
        """Execute a smooth joint/gripper ramp for contact-robust motion."""
        arm, gripper = ARMS[side]
        for i in range(1, steps + 1):
            alpha = i / steps
            action = hold_action(self.rv)
            action[arm] = (1.0 - alpha) * np.asarray(start_q) + alpha * np.asarray(end_q)
            action[gripper] = np.array([(1.0 - alpha) * start_gripper + alpha * end_gripper])
            self.execute(action, phase, side)

    def hold_until_pose(
        self,
        side: str,
        target_q: np.ndarray,
        gripper_command: float,
        target_pose: np.ndarray,
        phase: str,
        max_steps: int = 1200,
        pos_tol_m: float = 0.006,
        rot_tol_rad: float = 0.06,
        min_steps: int = 80,
        stable_steps: int = 20,
    ) -> dict[str, Any]:
        arm, gripper = ARMS[side]
        stable = 0
        last_pos = float("inf")
        last_rot = float("inf")
        for step in range(1, max_steps + 1):
            action = hold_action(self.rv)
            action[arm] = np.asarray(target_q).copy()
            action[gripper] = np.array([gripper_command], dtype=np.float64)
            self.execute(action, phase, side)
            tcp = self.rv.get_move_group(gripper).leaf_frame_to_world
            last_pos = float(np.linalg.norm(tcp[:3, 3] - target_pose[:3, 3]))
            last_rot = float(
                reach_mod.R.from_matrix(tcp[:3, :3] @ target_pose[:3, :3].T).magnitude()
            )
            if step >= min_steps and last_pos <= pos_tol_m and last_rot <= rot_tol_rad:
                stable += 1
                if stable >= stable_steps:
                    return {
                        "converged": True,
                        "steps": step,
                        "position_error_m": last_pos,
                        "rotation_error_rad": last_rot,
                    }
            else:
                stable = 0
        return {
            "converged": False,
            "steps": max_steps,
            "position_error_m": last_pos,
            "rotation_error_rad": last_rot,
        }

    def settle(self, steps=120):
        for _ in range(steps):
            self.execute(hold_action(self.rv), "settle", "none")


def snapshot_physics(env) -> np.ndarray:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.current_model, spec), dtype=np.float64)
    mujoco.mj_getState(env.current_model, env.current_data, state, spec)
    return state


def restore_physics(env, state: np.ndarray) -> None:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    # mj_setState alone leaves derived/contact solver buffers from the previous
    # rollout resident in MjData. Clear the whole data arena first, then restore
    # the complete integration state and recompute derived quantities. This is
    # the canonical deterministic reset boundary needed for contact replay.
    mujoco.mj_resetData(env.current_model, env.current_data)
    mujoco.mj_setState(env.current_model, env.current_data, state, spec)
    mujoco.mj_forward(env.current_model, env.current_data)
    # Reset position-controller targets to the restored joint state; otherwise
    # a failed probe's final target leaks into the next candidate.
    action = hold_action(env.current_robot.robot_view)
    env.current_robot.update_control(action)
    env.current_robot.compute_control()


def grasp_diagnostics(env, rv, side: str, obj, target_pose: np.ndarray) -> dict[str, Any]:
    data, model = env.current_data, env.current_model
    gripper = ARMS[side][1]
    tcp = rv.get_move_group(gripper).leaf_frame_to_world.copy()
    obj_root = obj.body_id
    robot_root = rv.base.root_body_id
    contacts = []
    for contact in data.contact:
        body1 = int(model.body_rootid[model.geom_bodyid[contact.geom1]])
        body2 = int(model.body_rootid[model.geom_bodyid[contact.geom2]])
        if obj_root not in (body1, body2):
            continue
        other = body2 if body1 == obj_root else body1
        contacts.append(
            {
                "geom1": model.geom(int(contact.geom1)).name,
                "geom2": model.geom(int(contact.geom2)).name,
                "other_is_robot": bool(other == robot_root),
                "distance": float(contact.dist),
            }
        )
    mg = rv.get_move_group(gripper)
    return {
        "tcp_position": tcp[:3, 3],
        "target_position": target_pose[:3, 3],
        "tcp_position_error_m": float(np.linalg.norm(tcp[:3, 3] - target_pose[:3, 3])),
        "tcp_rotation_error_rad": float(
            reach_mod.R.from_matrix(tcp[:3, :3] @ target_pose[:3, :3].T).magnitude()
        ),
        "gripper_joint_pos": mg.joint_pos.copy(),
        "gripper_inter_finger_dist_m": float(mg.inter_finger_dist),
        "gripper_ctrl": mg.ctrl.copy(),
        "object_position": obj.position.copy(),
        "object_contacts": contacts,
        "robot_object_contact_count": sum(item["other_is_robot"] for item in contacts),
    }


def execute_pick_place(
    recorder: Recorder, side: str, object_name: str, box_name: str, slot_y: float
):
    env, rv = recorder.env, recorder.rv
    obj = create_mlspaces_body(env.current_data, object_name)
    box = create_mlspaces_body(env.current_data, box_name)
    arm, gripper = ARMS[side]
    q0 = copy_qpos(rv)
    base = rv.base.pose.copy()

    # Use the official DROID grasp library and its full TCP orientation.  A
    # center-point/top-down pose is not a valid grasp merely because IK solves.
    # Select a library-valid object grasp, then independently solve a reachable
    # box-opening TCP pose.  Requiring the grasp's *world* orientation to stay
    # fixed during transport was over-constrained: a held rigid object may be
    # reoriented with the wrist while preserving its local grasp transform.
    robot = env.current_robot
    grasps = reach_mod.get_pickup_grasps(env, obj, include_flipped=True, grasp_libraries=["droid"])
    order = reach_mod.rank_grasps(
        rv, obj, grasps, gripper, env.current_robot.exp_config.policy_config
    )
    q0_rotation = rv.get_move_group(gripper).leaf_frame_to_world[:3, :3].copy()
    pregrasp_offset = float(env.current_robot.exp_config.policy_config.pregrasp_z_offset)
    # Build YAM-native vertical grasps from the compiled robot geometry rather
    # than assuming Franka/DROID TCP axes. The local closing axis comes from the
    # two finger bodies; the insertion axis comes from palm to grasp-site.
    model, data = env.current_model, env.current_data
    namespace = env.current_robot.exp_config.robot_config.robot_namespace
    left_finger_pos = data.xpos[model.body(f"{namespace}{side}_link_left_finger").id].copy()
    right_finger_pos = data.xpos[model.body(f"{namespace}{side}_link_right_finger").id].copy()
    palm_pos = data.xpos[model.body(f"{namespace}{side}_link_6").id].copy()
    tcp_pose = rv.get_move_group(gripper).leaf_frame_to_world.copy()
    closing_local = tcp_pose[:3, :3].T @ (right_finger_pos - left_finger_pos)
    closing_local /= np.linalg.norm(closing_local)
    insertion_local = tcp_pose[:3, :3].T @ (tcp_pose[:3, 3] - palm_pos)
    insertion_local -= closing_local * float(np.dot(insertion_local, closing_local))
    insertion_local /= np.linalg.norm(insertion_local)
    third_local = np.cross(closing_local, insertion_local)
    local_basis = np.column_stack([closing_local, insertion_local, third_local])
    object_center, _ = body_aabb(model, data, obj.body_id)
    native_candidates = []
    for yaw in np.linspace(0.0, np.pi, 8, endpoint=False):
        closing_world = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        insertion_world = np.array([0.0, 0.0, -1.0])
        third_world = np.cross(closing_world, insertion_world)
        world_basis = np.column_stack([closing_world, insertion_world, third_world])
        rotation = world_basis @ local_basis.T
        for z_offset in (-0.006, 0.0, 0.006):
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = object_center + np.array([0.0, 0.0, z_offset])
            native_candidates.append(pose)

    ranked_droid = [(int(index), grasps[index]) for index in order[:64]]
    # Keep the known left candidate as a fast probe. Do not prioritize Apple
    # candidate 150: it can pass the pre-plan repeat yet fail the stricter
    # post-plan repeat. The native scan deterministically reaches candidate 15,
    # which passed both post-plan repeats in v30.
    preferred = {
        ("left", "Candle_1"): ("droid_full_pose", 1252),
    }.get((side, object_name.rsplit("/", 1)[-1]))
    candidate_sets = []
    if preferred is not None:
        preferred_mode, preferred_index = preferred
        candidate_sets.append((preferred_mode, [(preferred_index, grasps[preferred_index])]))
    candidate_sets.extend(
        [
            ("yam_native_vertical", list(enumerate(native_candidates))),
            ("droid_full_pose", ranked_droid),
            ("yam_tcp_orientation", ranked_droid),
        ]
    )
    initial_state = snapshot_physics(env)
    probe_records = []
    grasp_result = None
    recorder.record_enabled = False
    try:
        # Every candidate starts from the exact same MuJoCo integration state
        # and must dynamically lift the object before formal recording.
        for orientation_mode, candidates in candidate_sets:
            for rank, (candidate_index, candidate_pose) in enumerate(candidates):
                restore_physics(env, initial_state)
                recorder.probe_trace = []
                recorder.probe_trace_enabled = True
                obj_start = obj.position.copy()
                grasp_pose = candidate_pose.copy()
                pregrasp_pose = grasp_pose.copy()
                if orientation_mode == "yam_native_vertical":
                    pregrasp_pose[2, 3] += 0.08
                else:
                    pregrasp_pose[:3, 3] -= pregrasp_offset * pregrasp_pose[:3, 2]
                if orientation_mode == "yam_tcp_orientation":
                    grasp_pose[:3, :3] = q0_rotation
                    pregrasp_pose[:3, :3] = q0_rotation
                pre_solution = reach_mod.solve_ik(robot, gripper, arm, pregrasp_pose, q0, base)
                grasp_solution = reach_mod.solve_ik(robot, gripper, arm, grasp_pose, q0, base)
                lift_pose = grasp_pose.copy()
                lift_pose[2, 3] += 0.12
                lift_solution = reach_mod.solve_ik(robot, gripper, arm, lift_pose, q0, base)
                if pre_solution is None or grasp_solution is None or lift_solution is None:
                    continue
                # IK queries can temporarily write robot/model state. The dynamic
                # probe must begin from the same clean integration state as formal
                # command replay and fresh reset, not from solver residue.
                restore_physics(env, initial_state)
                obj_start = obj.position.copy()
                recorder.probe_trace = []
                recorder.probe_trace_enabled = True
                pre_gate = recorder.hold_until_pose(
                    side,
                    np.asarray(pre_solution[arm]),
                    OPEN,
                    pregrasp_pose,
                    "probe_pregrasp",
                    max_steps=700,
                    min_steps=40,
                    stable_steps=10,
                )
                if not pre_gate["converged"]:
                    continue
                approach_gate = recorder.hold_until_pose(
                    side,
                    np.asarray(grasp_solution[arm]),
                    OPEN,
                    grasp_pose,
                    "probe_approach",
                    max_steps=700,
                    min_steps=40,
                    stable_steps=10,
                )
                if not approach_gate["converged"]:
                    continue
                grasp_q = np.asarray(grasp_solution[arm]).copy()
                lift_q = np.asarray(lift_solution[arm]).copy()
                # Abrupt position targets make marginal contacts chaotic. Close
                # and lift with explicit ramps, then settle at the final target;
                # the exact low-level sequence is still subject to clean-repeat.
                recorder.ramp(side, grasp_q, grasp_q, OPEN, CLOSED, 240, "probe_close_ramp")
                recorder.hold(side, grasp_q, CLOSED, 120, "probe_close_hold")
                close_diag = grasp_diagnostics(env, rv, side, obj, grasp_pose)
                recorder.ramp(side, grasp_q, lift_q, CLOSED, CLOSED, 360, "probe_lift_ramp")
                recorder.hold_until_pose(
                    side,
                    lift_q,
                    CLOSED,
                    lift_pose,
                    "probe_lift_hold",
                    max_steps=500,
                    pos_tol_m=0.008,
                    rot_tol_rad=0.08,
                    min_steps=40,
                    stable_steps=10,
                )
                lift_m = float(obj.position[2] - obj_start[2])
                probe = {
                    "orientation_mode": orientation_mode,
                    "candidate_index": int(candidate_index),
                    "candidate_rank": rank,
                    "lift_m": lift_m,
                    "robot_contact_count_after_close": close_diag["robot_object_contact_count"],
                    "gripper_width_after_close_m": close_diag["gripper_inter_finger_dist_m"],
                }
                probe_records.append(probe)
                recorder.probe_trace_enabled = False
                print("GRASP_CANDIDATE_PROBE=" + json.dumps(to_jsonable(probe)), flush=True)
                if lift_m >= 0.025:
                    # Accept only candidates whose exact low-level trace repeats
                    # from the clean integration state. This rejects successes
                    # caused by search/solver residue before formal recording.
                    successful_trace = [
                        ({k: np.asarray(v).copy() for k, v in action.items()}, phase, active_side)
                        for action, phase, active_side in recorder.probe_trace
                    ]
                    restore_physics(env, initial_state)
                    repeat_start_z = float(obj.position[2])
                    recorder.probe_trace_enabled = False
                    for action, phase, active_side in successful_trace:
                        recorder.execute(action, phase, active_side)
                    repeat_lift_m = float(obj.position[2] - repeat_start_z)
                    repeat_payload = {
                        "side": side,
                        "candidate_index": int(candidate_index),
                        "candidate_rank": rank,
                        "probe_lift_m": lift_m,
                        "clean_repeat_lift_m": repeat_lift_m,
                        "repeat_pass": bool(repeat_lift_m >= 0.025),
                    }
                    print(
                        "GRASP_CANDIDATE_REPEAT=" + json.dumps(to_jsonable(repeat_payload)),
                        flush=True,
                    )
                    if repeat_lift_m < 0.025:
                        continue
                    repeated_tcp_pose = rv.get_move_group(gripper).leaf_frame_to_world.copy()
                    repeated_object_pose = obj.pose.copy()
                    repeated_object_position = obj.position.copy()
                    repeated_qpos = copy_qpos(rv)
                    grasp_result = {
                        "success": True,
                        "orientation_mode": orientation_mode,
                        "candidate_index": int(candidate_index),
                        "candidate_rank": rank,
                        "pregrasp_pose": pregrasp_pose,
                        "grasp_pose": grasp_pose,
                        "lift_pose": lift_pose,
                        "pregrasp_solution": pre_solution,
                        "grasp_solution": grasp_solution,
                        "lift_solution": lift_solution,
                        "probe_lift_m": lift_m,
                        "probe_records": probe_records.copy(),
                        "formal_command_trace": successful_trace,
                        "clean_repeat_lift_m": repeat_lift_m,
                        "lifted_tcp_pose": repeated_tcp_pose,
                        "lifted_object_pose": repeated_object_pose,
                        "lifted_object_position": repeated_object_position,
                        "lifted_qpos": repeated_qpos,
                    }
                    break
            if grasp_result is not None:
                break
    finally:
        recorder.probe_trace_enabled = False
        restore_physics(env, initial_state)
        recorder.record_enabled = True
    if grasp_result is None:
        raise RuntimeError(
            f"no dynamically successful grasp for {side}/{object_name}; "
            f"dynamic_probes={probe_records}"
        )
    # Plan placement using the *held object's* measured offset from the TCP.
    # The former box-opening helper targeted the TCP itself to the box center,
    # which does not imply that the grasped object enters the box.
    model, data = env.current_model, env.current_data
    box_center, box_size = reach_mod.body_aabb(model, data, box.body_id)
    place_result = None
    seed_q = grasp_result["lifted_qpos"]
    # Preserve the validated left-arm q0 placement branch while using the
    # lifted-state seed for the right arm, whose placement IK otherwise
    # failed from the pre-grasp state. Physical release remains the gate.
    placement_seed = q0 if side == "left" else seed_q
    lifted_tcp = grasp_result["lifted_tcp_pose"]
    lifted_obj_pose = grasp_result["lifted_object_pose"]
    lifted_obj = grasp_result["lifted_object_position"]
    tcp_to_object = np.linalg.inv(lifted_tcp) @ lifted_obj_pose
    attempts = 0
    box_aabb_top_z = float(box_center[2] + box_size[2] / 2.0)
    # Box_24's full-body AABB includes tall flaps/walls and is not the opening
    # plane (v33 produced an impossible 1.453 m object target). Follow the
    # reviewed official opening-grid convention: box center + 4 cm is staging
    # over the usable interior. Search small offsets above that reference.
    box_opening_reference_z = float(box_center[2] + 0.04)
    # Releasing just above the opening is both reachable and physically valid:
    # the object must then fall into and become supported by the real box. A
    # TCP target at box_center can be unnecessarily below the arm workspace.
    # Prefer the arm's known-reachable neutral TCP orientation. The object may
    # rotate rigidly in the grasp; only its center must be aligned over the box.
    # Full object-orientation preservation drove the YAM into a wrist singularity.
    orientation_options = [q0_rotation, lifted_tcp[:3, :3]]
    local_object_offset = tcp_to_object[:3, 3]
    # Try the safest low, central drop first. The previous edge-first order put
    # Candle near the box rim: it was released but settled outside support.
    # Dynamic reach evidence shows the left arm saturates ~7 cm on its side of
    # box center. Prefer that arm-side opening region rather than an unreachable
    # center; strict physical support remains the final gate.
    # Target the arm-side interior of Box_24 directly. Measured loaded-arm
    # reach saturates near world y=-0.33 (left) / -0.57 (right); +/-0.125 m
    # relative to the shared box center lies inside its opening and avoids
    # dragging a held object laterally for dozens of slip-prone corrections.
    # Compensate the measured loaded-arm tracking offset in the planned object
    # target without moving the shared box or changing the grasp initial state.
    # Final acceptance remains physical support plus robot release.
    # v48 strictly validated the compensated left target. v38/v39 logs show
    # the executable right desired world y was about -0.514 m, placing it near
    # a singular reach boundary. Search that narrow neighbourhood at 5 mm
    # resolution instead of assuming one rounded offset.
    if side == "left":
        y_offsets = (-0.03, 0.0, slot_y, -slot_y)
    else:
        y_offsets = tuple(np.arange(-0.060, -0.101, -0.005)) + (0.0, slot_y, -slot_y)
    for release_height in (0.00, 0.02, 0.04, 0.06):
        for target_rotation in orientation_options:
            for x_fraction in (0.0, -0.15, 0.15):
                # slot_y and y_offsets are absolute lateral offsets in metres.
                for y_offset in y_offsets:
                    attempts += 1
                    desired_obj = np.array(
                        [
                            box_center[0] + x_fraction * box_size[0],
                            box_center[1] + y_offset,
                            box_opening_reference_z + release_height,
                        ]
                    )
                    place_pose = np.eye(4)
                    place_pose[:3, :3] = target_rotation
                    place_pose[:3, 3] = desired_obj - target_rotation @ local_object_offset
                    preplace_pose = place_pose.copy()
                    # Preserve the validated 8 cm left preplace. For the right arm,
                    # the held-object search repeatedly hit a wrist-singular boundary
                    # at +8 cm although the lower place neighbourhood is the relevant
                    # target. Use a bounded 4 cm vertical staging pose for right only.
                    preplace_dz = 0.08 if side == "left" else 0.04
                    preplace_pose[2, 3] += preplace_dz
                    clearance_pose = preplace_pose.copy()
                    clearance_pose[2, 3] += preplace_dz
                    q_preplace_result = reach_mod.solve_ik(
                        robot, gripper, arm, preplace_pose, placement_seed, base
                    )
                    q_place_result = reach_mod.solve_ik(
                        robot, gripper, arm, place_pose, placement_seed, base
                    )
                    # Clearance is optional and expensive; solve it only after the
                    # mandatory place/preplace pair exists.
                    q_clearance_result = None
                    if q_preplace_result is not None and q_place_result is not None:
                        q_clearance_result = reach_mod.solve_ik(
                            robot, gripper, arm, clearance_pose, placement_seed, base
                        )
                    if attempts % 12 == 0 or (
                        q_preplace_result is not None and q_place_result is not None
                    ):
                        print(
                            "PLACEMENT_SEARCH="
                            + json.dumps(
                                to_jsonable(
                                    {
                                        "side": side,
                                        "attempt": attempts,
                                        "release_height": release_height,
                                        "x_fraction": x_fraction,
                                        "y_offset": y_offset,
                                        "preplace_dz": preplace_dz,
                                        "preplace_ok": q_preplace_result is not None,
                                        "place_ok": q_place_result is not None,
                                        "clearance_ok": q_clearance_result is not None,
                                    }
                                )
                            ),
                            flush=True,
                        )
                    # A vertical clearance solution is optional. The strict physical
                    # release gate already requires box support and zero robot contact
                    # for two consecutive checks, and fails immediately if support is
                    # lost. Requiring an additional +16 cm loaded-arm pose rejected the
                    # previously executable right-arm drop neighbourhood. When vertical
                    # clearance is unreachable, separate only along the collision-tested
                    # place -> preplace reverse path; acceptance remains unchanged.
                    if q_preplace_result is not None and q_place_result is not None:
                        if q_clearance_result is None:
                            q_clearance_result = q_preplace_result
                            clearance_pose = preplace_pose.copy()
                        place_result = {
                            "success": True,
                            "attempts": attempts,
                            "first_pose": preplace_pose,
                            "second_pose": place_pose,
                            "first_solution": q_preplace_result,
                            "second_solution": q_place_result,
                            "clearance_pose": clearance_pose,
                            "clearance_solution": q_clearance_result,
                            "desired_object_position": desired_obj,
                            "release_height": release_height,
                            "box_aabb_top_z": box_aabb_top_z,
                            "box_opening_reference_z": box_opening_reference_z,
                            "target_orientation": target_rotation,
                        }
                        break
                if place_result is not None:
                    break
            if place_result is not None:
                break
        if place_result is not None:
            break
    if place_result is None:
        raise RuntimeError(f"no held-object-relative box placement for {side}/{object_name}")
    # Placement IK has side effects beyond the MuJoCo integration vector. Flush
    # that path, then require two consecutive clean replays *after* planning
    # before allowing the recorded formal replay. This is stricter than the
    # earlier pre-planning repeat and prevents planner-cache contamination.
    post_plan_repeats = []
    recorder.record_enabled = False
    for repeat_index in range(2):
        restore_physics(env, initial_state)
        repeat_start_z = float(obj.position[2])
        for action, phase, active_side in grasp_result["formal_command_trace"]:
            recorder.execute(action, phase, active_side)
        repeat_lift = float(obj.position[2] - repeat_start_z)
        post_plan_repeats.append(repeat_lift)
        print(
            "POST_PLAN_GRASP_REPEAT="
            + json.dumps(
                to_jsonable(
                    {
                        "side": side,
                        "repeat_index": repeat_index,
                        "lift_m": repeat_lift,
                        "pass": bool(repeat_lift >= 0.025),
                    }
                )
            ),
            flush=True,
        )
    recorder.record_enabled = True
    if any(value < 0.025 for value in post_plan_repeats):
        raise RuntimeError(
            f"post-placement-planning grasp repeat failed for {side}/{object_name}: "
            f"{post_plan_repeats}"
        )
    restore_physics(env, initial_state)

    # Formal capture is a clean-state rerun, not the probe trajectory. Replay
    # the exact low-level command sequence that passed the dynamic gate instead
    # of re-solving/reconstructing a contact-sensitive trajectory.
    formal_start_z = float(obj.position[2])
    for action, phase, active_side in grasp_result["formal_command_trace"]:
        recorder.execute(action, phase.removeprefix("probe_"), active_side)
    after_lift = obj.position.copy()
    lift_m = float(after_lift[2] - formal_start_z)
    approach_diag = grasp_diagnostics(env, rv, side, obj, grasp_result["grasp_pose"])
    close_diag = approach_diag
    lift_convergence = {
        "converged": True,
        "steps": len(grasp_result["formal_command_trace"]),
        "position_error_m": float(
            np.linalg.norm(
                rv.get_move_group(gripper).leaf_frame_to_world[:3, 3]
                - grasp_result["lift_pose"][:3, 3]
            )
        ),
        "rotation_error_rad": float(
            reach_mod.R.from_matrix(
                rv.get_move_group(gripper).leaf_frame_to_world[:3, :3]
                @ grasp_result["lift_pose"][:3, :3].T
            ).magnitude()
        ),
    }
    if lift_m < 0.025:
        failure = {
            "side": side,
            "object_name": object_name,
            "lift_m": lift_m,
            "approach": approach_diag,
            "close": close_diag,
            "lift_convergence": lift_convergence,
            "after_lift_position": after_lift,
            "grasp_candidate_index": grasp_result["candidate_index"],
            "grasp_candidate_rank": grasp_result["candidate_rank"],
        }
        print("GRASP_FAILURE_DIAGNOSTIC=" + json.dumps(to_jsonable(failure)), flush=True)
        raise RuntimeError(f"{side} grasp/lift failed for {object_name}: lift={lift_m:.6f} m")

    q_lift = np.asarray(grasp_result["lift_solution"][arm]).copy()
    q_preplace = np.asarray(place_result["first_solution"][arm]).copy()
    q_place = np.asarray(place_result["second_solution"][arm]).copy()
    q_clearance = np.asarray(place_result["clearance_solution"][arm]).copy()
    # Preserve the grasp through transport with bounded joint ramps instead of
    # discontinuous position targets; keep the object closed until fully inside.
    recorder.ramp(side, q_lift, q_preplace, CLOSED, CLOSED, 600, "transport_ramp")
    recorder.hold(side, q_preplace, CLOSED, 120, "transport_hold")
    recorder.ramp(side, q_preplace, q_place, CLOSED, CLOSED, 240, "place_ramp")
    recorder.hold(side, q_place, CLOSED, 100, "place_hold")
    # Contact dynamics and finite joint tracking can leave the held object's
    # actual center several centimetres from the planned box-center target.
    # Before opening, close the oracle loop on the measured held-object center.
    # The resulting correction remains a fixed low-level command in the saved
    # trace, so fresh replay does not use privileged feedback.
    # The loaded object is closest to the selected arm-side interior target at
    # the first collision-tested place pose. Closed-gripper correction loops
    # made the object slip downward and diverge, so release from this pose and
    # let the strict physical support/contact gate decide validity.
    release_q = q_place.copy()
    release_waypoints = [q_place.copy()]
    final_alignment = np.asarray(place_result["desired_object_position"]) - obj.position.copy()
    final_alignment[2] = 0.0
    print(
        "PRE_RELEASE_ALIGNMENT_FINAL="
        + json.dumps(
            to_jsonable(
                {
                    "side": side,
                    "delta_m": final_alignment,
                    "xy_error_m": float(np.linalg.norm(final_alignment[:2])),
                }
            )
        ),
        flush=True,
    )
    # Do not substitute an arbitrary center-distance threshold for the real
    # receptacle gate. The left wrist saturates near y=-0.37 m, but the held
    # object may still be over the physical opening. Release and let the strict
    # objects_on_receptacle + robot-release test decide; this does not lower the
    # task success criterion.
    recorder.ramp(side, release_q, release_q, CLOSED, OPEN, 180, "release_ramp")
    # Releasing is itself a strict behavioral gate, not a fixed-duration guess.
    # Keep the wrist stationary and fingers fully open until the object is both
    # physically supported by the box and free of robot contact for consecutive
    # checks. The resulting bounded command sequence is recorded verbatim and
    # replayed without oracle feedback.
    release_gate = None
    release_stable_checks = 0
    release_gate_q = release_q.copy()
    support_seen = False
    # Once the object is supported, residual open-finger contact is broken by
    # bounded micro-steps along the already collision-tested reverse placement
    # path. Every command is recorded; fresh replay executes it without oracle
    # feedback. Losing box support after first contact is an immediate failure.
    for release_check in range(60):
        recorder.hold(side, release_gate_q, OPEN, 40, "release_hold")
        release_gate = strict_success(recorder.task, [object_name], box_name)["per_object"][
            object_name
        ]
        print(
            "RELEASE_GATE="
            + json.dumps(
                to_jsonable(
                    {
                        "side": side,
                        "check": release_check,
                        **release_gate,
                    }
                )
            ),
            flush=True,
        )
        if support_seen and not release_gate["supported_by_box"]:
            raise RuntimeError(
                f"box support lost during release separation for {side}/{object_name}: {release_gate}"
            )
        support_seen = support_seen or release_gate["supported_by_box"]
        if release_gate["pass"]:
            release_stable_checks += 1
            if release_stable_checks >= 2:
                break
        else:
            release_stable_checks = 0
            if release_gate["supported_by_box"] and not release_gate["released_from_robot"]:
                # Follow a vertical two-segment retreat: place -> preplace ->
                # clearance. Returning toward q_lift introduced horizontal drag
                # and correctly failed the box-support gate in v53.
                separation_target = q_preplace if release_check < 30 else q_clearance
                next_q = release_gate_q + 0.10 * (separation_target - release_gate_q)
                recorder.ramp(
                    side, release_gate_q, next_q, OPEN, OPEN, 40, "release_separation_microstep"
                )
                release_gate_q = next_q
    if release_stable_checks < 2:
        raise RuntimeError(f"strict release gate failed for {side}/{object_name}: {release_gate}")
    # Retreat with the gripper open along the reverse of the collision-tested
    # placement path. This is deterministic, remains in the saved trace, and
    # avoids a fresh IK solve at a joint-limit-sensitive release endpoint.
    retreat_q = release_gate_q.copy()
    for waypoint_index, waypoint_q in enumerate(reversed(release_waypoints[:-1])):
        recorder.ramp(
            side, retreat_q, waypoint_q, OPEN, OPEN, 180, f"post_release_unwind_{waypoint_index}"
        )
        retreat_q = waypoint_q.copy()
    recorder.ramp(side, retreat_q, q_preplace, OPEN, OPEN, 240, "post_release_retreat")
    recorder.hold(side, q_preplace, OPEN, 100, "post_release_retreat_hold")
    recorder.settle(260)
    return {
        "lift_m": lift_m,
        "grasp_candidate_index": grasp_result["candidate_index"],
        "grasp_candidate_rank": grasp_result["candidate_rank"],
        "box_candidate_attempts": place_result["attempts"],
    }


def append_oracle_group(
    h5_path: Path,
    recorder: Recorder,
    layout: dict[str, Any],
    final: dict[str, Any],
    provenance: str,
):
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(h5_path, "a") as f:
        traj = f["traj_0"]
        root = traj.create_group("mimicgen_yam")
        root.attrs["schema_version"] = "0.3"
        root.attrs["source_provenance"] = provenance
        root.attrs["coordination_mode"] = "sequential"
        root.attrs["strict_2of2_success"] = bool(final["success"])
        root.attrs["control_stride"] = CONTROL_STRIDE
        root.create_dataset(
            "sim_step", data=np.asarray([r["sim_step"] for r in recorder.records], dtype=np.int64)
        )
        root.create_dataset(
            "phase", data=np.asarray([r["phase"] for r in recorder.records], dtype=string_dtype)
        )
        root.create_dataset(
            "active_side",
            data=np.asarray([r["active_side"] for r in recorder.records], dtype=string_dtype),
        )
        root.create_dataset(
            "left_tcp_pose", data=np.stack([r["left_tcp"] for r in recorder.records])
        )
        root.create_dataset(
            "right_tcp_pose", data=np.stack([r["right_tcp"] for r in recorder.records])
        )
        for name in recorder.object_names:
            root.create_dataset(
                f"object_pose/{name}",
                data=np.stack([r["object_poses"][name] for r in recorder.records]),
            )
        commands = root.create_group("replay_commands")
        for key in ("left_arm", "right_arm", "left_gripper", "right_gripper"):
            commands.create_dataset(
                key, data=np.stack([a[key] for a in recorder.commands]), compression="gzip"
            )
        root.create_dataset("layout_json", data=json.dumps(to_jsonable(layout)), dtype=string_dtype)
        root.create_dataset(
            "final_success_json", data=json.dumps(to_jsonable(final)), dtype=string_dtype
        )


def initialize_episode_caches(task):
    task.episode_step_count = 0
    task.action_cache = []
    task.observation_cache = []
    task.reward_cache = []
    task.terminal_cache = []
    task.truncated_cache = []
    task.success_cache = []
    task.frozen_config = {}


def sample_task(args):
    # Parameterize the already-reviewed additive sampler without changing the
    # upstream task/sampler files or the previously accepted Gate-1D artifacts.
    reach_mod.PICKUP_UIDS = PICKUP_UIDS
    init_mod.PICKUP_UIDS = PICKUP_UIDS
    return init_mod.sample_strict_tabletop_task(
        args.house_index,
        args.seed,
        args.initialization_max_attempts,
        output_dir=str(args.output_dir),
        policy_dt_ms=40,
        task_horizon=100000,
    )


def settle_relocated_layout(task, object_names: list[str], box_name: str, steps: int = 600):
    probe = Recorder(task, object_names, box_name, record_enabled=False)
    probe.settle(steps)
    return {
        name: create_mlspaces_body(task.env.current_data, name).position.copy()
        for name in object_names + [box_name]
    }


def run_capture(args):
    sampler = task = recorder = None
    try:
        sampler, task, attempts = sample_task(args)
        initialize_episode_caches(task)
        layout = place_for_arm_assignment(task, sampler)
        env = task.env
        object_names = [
            layout["uid_to_name"][layout["assignment"][side]] for side in ("left", "right")
        ]
        settled_positions = settle_relocated_layout(task, object_names, layout["box_name"])
        layout["settled_world_positions"] = settled_positions
        layout["settle_steps_before_planning"] = 600
        recorder = Recorder(task, object_names, layout["box_name"])
        execution = {}
        for side, slot in (("left", +0.055), ("right", -0.055)):
            uid = layout["assignment"][side]
            execution[side] = execute_pick_place(
                recorder, side, layout["uid_to_name"][uid], layout["box_name"], slot
            )
        final = strict_success(task, object_names, layout["box_name"])
        if not final["success"]:
            raise RuntimeError(f"strict 2/2 failed before history materialization: {final}")
        recorder.materialize_history()
        history = task.get_history()
        prepared = prepare_episode_for_saving(
            history,
            task.sensor_suite,
            fps=sampler.config.fps,
            save_dir=str(args.output_dir),
            episode_idx=0,
        )
        if prepared is None:
            raise RuntimeError("prepare_episode_for_saving returned None")
        h5_path = Path(save_trajectories([prepared], str(args.output_dir), fps=sampler.config.fps))
        append_oracle_group(h5_path, recorder, layout, final, "scripted_oracle_planner")
        manifest = {
            "gate": "1D-3-scripted-expert",
            "hdf5": h5_path,
            "house_index": args.house_index,
            "base_seed": args.seed,
            "attempts": attempts,
            "layout": layout,
            "execution": execution,
            "strict_success": final,
            "policy_samples": len(recorder.records),
            "low_level_commands": len(recorder.commands),
            "source_provenance": "scripted_oracle_planner",
            "evidence_boundary": "One scripted-expert source candidate; not MimicGen expansion, policy learning, or unseen-reset policy success.",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "capture_manifest.json").write_text(
            json.dumps(to_jsonable(manifest), indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps(to_jsonable(manifest), indent=2, ensure_ascii=False), flush=True)
        if not final["success"]:
            raise SystemExit(2)
        return h5_path
    except Exception as exc:
        # Diagnostic-only failure video: preserve the exact executed physics trace
        # without changing any lift, release, strict-success, or replay gate.
        if recorder is not None and recorder.history_states:
            try:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                recorder.materialize_history()
                failure_history = task.get_history()
                prepare_episode_for_saving(
                    failure_history,
                    task.sensor_suite,
                    fps=sampler.config.fps,
                    save_dir=str(args.output_dir),
                    episode_idx=0,
                    save_file_suffix="_failure_diagnostic",
                )
                (args.output_dir / "failure_diagnostic.json").write_text(
                    json.dumps(
                        {"error_type": type(exc).__name__, "error": str(exc)},
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                print(f"FAILURE_DIAGNOSTIC_VIDEO_SAVED={args.output_dir}", flush=True)
            except Exception as diagnostic_exc:
                print(f"FAILURE_DIAGNOSTIC_VIDEO_ERROR={diagnostic_exc!r}", flush=True)
        raise
    finally:
        if sampler is not None:
            sampler.close()


def run_replay(args, h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        root = f["traj_0/mimicgen_yam"]
        commands = {
            key: root[f"replay_commands/{key}"][...]
            for key in ("left_arm", "right_arm", "left_gripper", "right_gripper")
        }
        source_layout = json.loads(
            root["layout_json"][()].decode()
            if isinstance(root["layout_json"][()], bytes)
            else root["layout_json"][()]
        )
        expected_final = {
            name: root[f"object_pose/{name}"][-1, :3] for name in root["object_pose"].keys()
        }
    sampler = task = None
    try:
        sampler, task, attempts = sample_task(args)
        layout = place_for_arm_assignment(task, sampler)
        object_names = [
            layout["uid_to_name"][layout["assignment"][side]] for side in ("left", "right")
        ]
        settle_relocated_layout(task, object_names, layout["box_name"], steps=600)
        reset_errors = {}
        for side in ("left", "right"):
            uid = layout["assignment"][side]
            name = layout["uid_to_name"][uid]
            live = create_mlspaces_body(task.env.current_data, name).position.copy()
            source = np.asarray(source_layout["settled_world_positions"][name])
            reset_errors[uid] = float(np.linalg.norm(live - source))
        lengths = {k: len(v) for k, v in commands.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"command length mismatch: {lengths}")
        robot = task.env.current_robot
        for index in range(next(iter(lengths.values()))):
            action = {key: commands[key][index] for key in commands}
            robot.update_control(action)
            robot.compute_control()
            task.env.step(1)
        final = strict_success(task, object_names, layout["box_name"])
        final_errors = {
            name: float(
                np.linalg.norm(
                    create_mlspaces_body(task.env.current_data, name).position
                    - expected_final[name]
                )
            )
            for name in object_names
        }
        payload = {
            "fresh_deterministic_reset": True,
            "attempts": attempts,
            "reset_errors_m": reset_errors,
            "reset_match": all(v <= 1e-6 for v in reset_errors.values()),
            "command_count": next(iter(lengths.values())),
            "strict_success": final,
            "final_position_errors_m": final_errors,
            "final_match": all(v <= 0.03 for v in final_errors.values()),
        }
        payload["replay_gate_pass"] = bool(
            payload["reset_match"] and final["success"] and payload["final_match"]
        )
        (args.output_dir / "replay_result.json").write_text(
            json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), flush=True)
        if not payload["replay_gate_pass"]:
            raise SystemExit(2)
    finally:
        if sampler is not None:
            sampler.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--house-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--initialization-max-attempts", type=int, default=8)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = run_capture(args)
    if args.replay:
        run_replay(args, h5_path)


if __name__ == "__main__":
    main()
