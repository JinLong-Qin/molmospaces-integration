"""Constrained FloorPlan1 island initialization for bimanual browser teleop.

Provenance is intentionally explicit:
- scene geometry: official MolmoSpaces iTHOR FloorPlan1;
- placement flow: official PackingTaskSampler / PickTaskSampler hooks;
- additive adaptation: two fixed pickup assets, island-only support filtering, and
  a post-placement wooden-platform edge alignment with collision validation.

This is not an untouched upstream task preset.  It rejects shelf/cabinet/counter
candidates and lets the official retry loop continue with another island candidate.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from check_dual_object_reachability import (
    BimanualYamPackingConfig,
    DualPickupDiagnosticSampler,
    PICKUP_UIDS,
    assign_sides,
    build_config as build_diagnostic_config,
    copy_qpos,
    diagnose_object_position_only,
    solve_box_opening_grid,
    to_jsonable,
)
from molmo_spaces.tasks.task_sampler_errors import (
    HouseInvalidForTask,
    RetriableError,
    RobotPlacementError,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb, body_base_pos
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom
from molmo_spaces.utils.pose import pose_mat_to_7d

ISLAND_SUPPORT_BODY_PREFIX = "standardislandheight_"
ISLAND_TOP_Z_RANGE = (0.95, 1.20)
MIN_FACING_COSINE = 0.70
WORKSPACE_DISTANCE_RANGE_M = (0.25, 1.50)
MAX_WORKSPACE_DIAMETER_M = 1.20
TARGET_PLATFORM_ISLAND_GAP_M = 0.005
MAX_PLATFORM_ISLAND_GAP_ERROR_M = 0.003
MIN_PLATFORM_ISLAND_LATERAL_OVERLAP_M = 0.20
MAX_PLATFORM_ALIGNMENT_SHIFT_M = 0.30


def _geom_xy_corners(model, data, geom_id: int) -> np.ndarray:
    """Return exact world-XY footprint corners for a box geom."""
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX.value:
        raise ValueError(f"Expected box geom, got type={model.geom_type[geom_id]}")
    half_size = np.asarray(model.geom_size[geom_id], dtype=float)
    local = np.asarray(
        [
            [sx * half_size[0], sy * half_size[1], sz * half_size[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    world = local @ rotation.T + np.asarray(data.geom_xpos[geom_id], dtype=float)
    return world[:, :2]


def _geom_xy_vertices(model, data, geom_id: int) -> np.ndarray:
    """Return world-XY vertices for a visible box or mesh geom."""
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX.value:
        return np.unique(_geom_xy_corners(model, data, geom_id), axis=0)
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH.value:
        mesh_id = int(model.geom_dataid[geom_id])
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=float)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        world = local @ rotation.T + np.asarray(data.geom_xpos[geom_id], dtype=float)
        return world[:, :2]
    raise ValueError(f"Unsupported footprint geom type={geom_type}")


def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """Compute a 2-D convex hull in counter-clockwise order."""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(pts) < 3:
        raise ValueError("Need at least three unique points for a footprint hull")

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _forward_ray_near_edge(
    footprint_xy: np.ndarray,
    origin_xy: np.ndarray,
    forward_xy: np.ndarray,
    right_xy: np.ndarray,
) -> float:
    """Distance from origin to the first footprint boundary hit along +forward."""
    relative = np.asarray(footprint_xy, dtype=float) - np.asarray(origin_xy, dtype=float)
    polygon = np.column_stack((relative @ forward_xy, relative @ right_xy))
    hits = []
    for index, p0 in enumerate(polygon):
        p1 = polygon[(index + 1) % len(polygon)]
        s0, s1 = float(p0[1]), float(p1[1])
        if abs(s0) < 1e-10:
            hits.append(float(p0[0]))
        if (s0 < 0.0 < s1) or (s1 < 0.0 < s0):
            alpha = -s0 / (s1 - s0)
            hits.append(float(p0[0] + alpha * (p1[0] - p0[0])))
    positive_hits = [hit for hit in hits if hit > 0.0]
    if not positive_hits:
        raise ValueError("Robot forward ray does not intersect the island footprint")
    return min(positive_hits)


def _nearest_suitable_island_edge(
    footprint_xy: np.ndarray,
    robot_xy: np.ndarray,
    platform_width_m: float,
) -> dict[str, Any]:
    """Select the nearest exterior hull edge that can contain the full platform width."""
    polygon = np.asarray(footprint_xy, dtype=float)
    candidates: list[dict[str, Any]] = []
    half_width = platform_width_m / 2.0
    for index, p0 in enumerate(polygon):
        p1 = polygon[(index + 1) % len(polygon)]
        edge_vec = p1 - p0
        edge_length = float(np.linalg.norm(edge_vec))
        if edge_length < platform_width_m:
            continue
        tangent = edge_vec / edge_length
        inward = np.asarray([-tangent[1], tangent[0]], dtype=float)
        raw_along = float(np.dot(robot_xy - p0, tangent))
        along = float(np.clip(raw_along, half_width, edge_length - half_width))
        edge_center = p0 + along * tangent
        # Convex hull is counter-clockwise, so interior is on the left side.
        # The robot must be outside this edge before alignment.
        if float(np.dot(robot_xy - edge_center, inward)) >= 0.0:
            continue
        candidates.append(
            {
                "index": index,
                "p0": p0,
                "p1": p1,
                "length_m": edge_length,
                "tangent": tangent,
                "inward": inward,
                "edge_center": edge_center,
                "distance_m": float(np.linalg.norm(robot_xy - edge_center)),
            }
        )
    if not candidates:
        raise ValueError("No exterior island edge can contain the full platform width")
    return min(candidates, key=lambda item: item["distance_m"])


def _project_interval(points_xy: np.ndarray, axis_xy: np.ndarray) -> tuple[float, float]:
    axis = np.asarray(axis_xy, dtype=float)
    axis /= np.linalg.norm(axis)
    values = np.asarray(points_xy, dtype=float) @ axis
    return float(values.min()), float(values.max())


class FloorPlan1IslandDualPickupSampler(DualPickupDiagnosticSampler):
    """Official placement flow plus island-only support and platform alignment."""

    def __init__(self, config: BimanualYamPackingConfig) -> None:
        super().__init__(config)
        self.accepted_initialization: dict[str, Any] | None = None
        self.rejected_initializations: list[dict[str, Any]] = []
        self._candidate_support_geom_id: int | None = None
        self._candidate_support: dict[str, Any] | None = None

    def _support_record(self, env, object_name: str) -> dict[str, Any] | None:
        om = env.object_managers[env.current_batch_index]
        body_id = om.get_object_body_id(object_name)
        geom_id = get_supporting_geom(env.current_data, body_id)
        if geom_id is None:
            return None
        model, data = env.current_model, env.current_data
        support_body_id = int(model.geom_bodyid[geom_id])
        return {
            "object_name": object_name,
            "object_base_pos": body_base_pos(data, body_id).copy(),
            "geom_id": int(geom_id),
            "geom_name": model.geom(geom_id).name,
            "body_id": support_body_id,
            "body_name": model.body(support_body_id).name,
            "geom_pos": np.asarray(data.geom_xpos[geom_id]).copy(),
            "geom_size": np.asarray(model.geom_size[geom_id]).copy(),
        }

    @staticmethod
    def _is_allowed_island_support(record: dict[str, Any] | None) -> bool:
        if record is None:
            return False
        body_name = record["body_name"] or ""
        top_z = float(record["geom_pos"][2])
        return body_name.startswith(ISLAND_SUPPORT_BODY_PREFIX) and (
            ISLAND_TOP_Z_RANGE[0] <= top_z <= ISLAND_TOP_Z_RANGE[1]
        )

    def _get_scene_objects(self, env, mass_limit=100):
        candidates = super()._get_scene_objects(env, mass_limit=mass_limit)
        accepted = []
        rejected = []
        for candidate in candidates:
            support = self._support_record(env, candidate.name)
            if self._is_allowed_island_support(support):
                accepted.append(candidate)
            else:
                rejected.append(
                    {
                        "candidate": candidate.name,
                        "reason": "not_floorplan1_island_support",
                        "support": support,
                    }
                )
        self.rejected_initializations.extend(rejected)
        if not accepted:
            raise HouseInvalidForTask(
                "FloorPlan1 has no pickup candidates on the allowed standardislandheight support"
            )
        return accepted

    def _on_candidate_selected(
        self,
        env,
        reference_obj_name: str,
        reference_obj_id: int,
        supporting_geom_id: int,
    ) -> bool:
        support = self._support_record(env, reference_obj_name)
        if not self._is_allowed_island_support(support):
            self.rejected_initializations.append(
                {
                    "candidate": reference_obj_name,
                    "reason": "candidate_support_changed_or_not_island",
                    "support": support,
                }
            )
            raise ValueError("candidate is not on the allowed FloorPlan1 island")
        self._candidate_support_geom_id = int(supporting_geom_id)
        self._candidate_support = support
        return super()._on_candidate_selected(
            env, reference_obj_name, reference_obj_id, supporting_geom_id
        )

    def _reject(self, reason: str, details: dict[str, Any]) -> None:
        record = {"reason": reason, **details}
        self.rejected_initializations.append(record)
        raise RobotPlacementError(reason)

    def _align_platform_edge_to_island(self, env, robot_view) -> dict[str, Any]:
        """Align the full wooden platform edge parallel to the nearest island edge."""
        model, data = env.current_model, env.current_data
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_0/base")
        if base_body_id < 0:
            self._reject("robot_base_body_not_found", {"body_name": "robot_0/base"})

        expected_half_size = np.asarray(self.config.robot_config.base_size, dtype=float) / 2.0
        platform_candidates = []
        geom_start = int(model.body_geomadr[base_body_id])
        geom_end = geom_start + int(model.body_geomnum[base_body_id])
        for geom_id in range(geom_start, geom_end):
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX.value:
                continue
            if np.allclose(
                np.asarray(model.geom_size[geom_id], dtype=float),
                expected_half_size,
                atol=1e-6,
                rtol=0.0,
            ):
                platform_candidates.append(geom_id)
        if len(platform_candidates) != 1:
            self._reject(
                "wooden_platform_geom_not_unique",
                {
                    "base_body_id": base_body_id,
                    "expected_half_size": expected_half_size,
                    "platform_candidates": platform_candidates,
                },
            )
        platform_geom_id = platform_candidates[0]

        visible_island_geom_ids = []
        island_vertices = []
        support_body_id = int(self._candidate_support["body_id"])
        geom_start = int(model.body_geomadr[support_body_id])
        geom_end = geom_start + int(model.body_geomnum[support_body_id])
        for geom_id in range(geom_start, geom_end):
            if int(model.geom_group[geom_id]) >= 3:
                continue
            try:
                vertices = _geom_xy_vertices(model, data, geom_id)
            except ValueError:
                continue
            visible_island_geom_ids.append(geom_id)
            island_vertices.append(vertices)
        if not island_vertices:
            self._reject(
                "island_visible_footprint_not_found",
                {"support_body_id": support_body_id},
            )
        island_footprint = _convex_hull_xy(np.concatenate(island_vertices, axis=0))

        base_pose_before = robot_view.base.pose.copy()
        platform_depth_m = float(self.config.robot_config.base_size[0])
        platform_width_m = float(self.config.robot_config.base_size[1])
        try:
            selected_edge = _nearest_suitable_island_edge(
                island_footprint,
                np.asarray(base_pose_before[:2, 3], dtype=float),
                platform_width_m,
            )
        except ValueError as exc:
            self._reject("suitable_island_edge_not_found", {"error": str(exc)})

        forward = np.asarray(selected_edge["inward"], dtype=float)
        local_y = -np.asarray(selected_edge["tangent"], dtype=float)
        desired_base_xy = np.asarray(selected_edge["edge_center"], dtype=float) - forward * (
            platform_depth_m / 2.0 + TARGET_PLATFORM_ISLAND_GAP_M
        )
        translation_xy = desired_base_xy - np.asarray(base_pose_before[:2, 3], dtype=float)
        translation_norm = float(np.linalg.norm(translation_xy))
        if translation_norm > MAX_PLATFORM_ALIGNMENT_SHIFT_M:
            self._reject(
                "platform_alignment_shift_too_large",
                {
                    "target_gap_m": TARGET_PLATFORM_ISLAND_GAP_M,
                    "required_translation_xy_m": translation_xy,
                    "required_translation_norm_m": translation_norm,
                    "max_translation_m": MAX_PLATFORM_ALIGNMENT_SHIFT_M,
                    "selected_edge": selected_edge,
                },
            )

        base_pose_after = base_pose_before.copy()
        base_pose_after[:2, 0] = forward
        base_pose_after[:2, 1] = local_y
        base_pose_after[:2, 3] = desired_base_xy
        robot_view.base.pose = base_pose_after
        mujoco.mj_forward(model, data)

        platform_points_after = _geom_xy_corners(model, data, platform_geom_id)
        platform_forward = _project_interval(platform_points_after, forward)
        platform_lateral = _project_interval(platform_points_after, local_y)
        edge_forward = float(np.dot(selected_edge["edge_center"], forward))
        edge_lateral = _project_interval(
            np.asarray([selected_edge["p0"], selected_edge["p1"]]), local_y
        )
        final_gap = edge_forward - platform_forward[1]
        lateral_overlap = min(platform_lateral[1], edge_lateral[1]) - max(
            platform_lateral[0], edge_lateral[0]
        )
        gap_error = abs(final_gap - TARGET_PLATFORM_ISLAND_GAP_M)
        collision = env.check_robot_collision_in_current_pose()
        if (
            gap_error > MAX_PLATFORM_ISLAND_GAP_ERROR_M
            or lateral_overlap < MIN_PLATFORM_ISLAND_LATERAL_OVERLAP_M
            or collision
        ):
            self._reject(
                "platform_edge_alignment_failed",
                {
                    "target_gap_m": TARGET_PLATFORM_ISLAND_GAP_M,
                    "final_gap_m": final_gap,
                    "gap_error_m": gap_error,
                    "max_gap_error_m": MAX_PLATFORM_ISLAND_GAP_ERROR_M,
                    "lateral_overlap_m": lateral_overlap,
                    "min_lateral_overlap_m": MIN_PLATFORM_ISLAND_LATERAL_OVERLAP_M,
                    "robot_collision": collision,
                    "base_pose_before": base_pose_before,
                    "base_pose_after": base_pose_after,
                },
            )

        self.config.task_config.robot_base_pose = pose_mat_to_7d(base_pose_after).tolist()
        return {
            "platform_geom_id": platform_geom_id,
            "platform_geom_name": model.geom(platform_geom_id).name,
            "support_geom_id": int(self._candidate_support_geom_id),
            "support_geom_name": model.geom(int(self._candidate_support_geom_id)).name,
            "edge_reference": "nearest full-width edge of full visible island mesh footprint",
            "visible_island_geom_ids": visible_island_geom_ids,
            "selected_island_edge": selected_edge,
            "target_gap_m": TARGET_PLATFORM_ISLAND_GAP_M,
            "translation_xy_m": translation_xy,
            "translation_norm_m": translation_norm,
            "final_gap_m": final_gap,
            "gap_error_m": gap_error,
            "lateral_overlap_m": lateral_overlap,
            "robot_collision": collision,
            "base_pose_before": base_pose_before,
            "base_pose_after": base_pose_after,
            "pass": True,
        }

    def _sample_and_place_robot(self, env) -> None:
        # This is the upstream placement implementation, including
        # env.place_robot_near(..., face_target=True).
        super()._sample_and_place_robot(env)
        mujoco.mj_forward(env.current_model, env.current_data)

        om = env.object_managers[env.current_batch_index]
        robot = env.current_robot
        robot_view = robot.robot_view

        if len(self._added_pickup_names) != 2:
            self._reject(
                "expected_exactly_two_added_pickups",
                {"pickup_names": list(self._added_pickup_names)},
            )
        if set(self._added_pickup_uids) != set(PICKUP_UIDS):
            self._reject(
                "unexpected_pickup_uids",
                {"pickup_uids": list(self._added_pickup_uids)},
            )

        uid_to_name = dict(zip(self._added_pickup_uids, self._added_pickup_names, strict=True))
        box_name = self.config.task_config.place_receptacle_name
        if not box_name:
            self._reject("missing_place_receptacle", {})

        objects_by_uid = {uid: om.get_object_by_name(name) for uid, name in uid_to_name.items()}
        box = om.get_object_by_name(box_name)
        workspace_positions = [obj.position.copy() for obj in objects_by_uid.values()]
        workspace_positions.append(box.position.copy())
        workspace_positions = np.asarray(workspace_positions, dtype=float)
        support_body_id = int(self._candidate_support["body_id"])
        support_center, support_size = body_aabb(
            env.current_model, env.current_data, support_body_id
        )
        support_top_z = float(support_center[2] + support_size[2] / 2.0)
        placed_objects = {
            **{uid_to_name[uid]: obj for uid, obj in objects_by_uid.items()},
            box_name: box,
        }
        placement_aabb = {}
        for name, obj in placed_objects.items():
            center, size = body_aabb(env.current_model, env.current_data, obj.body_id)
            bottom_z = float(center[2] - size[2] / 2.0)
            placement_aabb[name] = {
                "center": center,
                "size": size,
                "bottom_z": bottom_z,
                "bottom_to_support_top_m": bottom_z - support_top_z,
            }
        max_abs_bottom_delta_m = float(
            max(abs(record["bottom_to_support_top_m"]) for record in placement_aabb.values())
        )
        # The official placement hooks pass this exact island geom id to
        # place_object_near for both pickups and the box, then enforce a bottom-z
        # consistency check.  get_supporting_geom() can return None immediately
        # after placement because it is contact-derived, so do not reinterpret
        # that transient None as a different support.
        placement_support_pass = (
            self._candidate_support_geom_id is not None and max_abs_bottom_delta_m <= 0.03
        )
        if not placement_support_pass:
            self._reject(
                "official_same_support_placement_height_check_failed",
                {
                    "placement_support_geom_id": self._candidate_support_geom_id,
                    "support_top_z": support_top_z,
                    "placement_aabb": placement_aabb,
                    "max_abs_bottom_delta_m": max_abs_bottom_delta_m,
                    "max_allowed_bottom_delta_m": 0.03,
                },
            )

        edge_alignment = self._align_platform_edge_to_island(env, robot_view)
        assignment, base_positions = assign_sides(robot_view, objects_by_uid)
        workspace_center = workspace_positions.mean(axis=0)
        workspace_diameter = float(
            max(
                np.linalg.norm(a[:2] - b[:2])
                for a in workspace_positions
                for b in workspace_positions
            )
        )

        base_pose = robot_view.base.pose.copy()
        base_xy = base_pose[:2, 3]
        to_workspace = workspace_center[:2] - base_xy
        workspace_distance = float(np.linalg.norm(to_workspace))
        robot_forward = base_pose[:2, 0]
        forward_norm = float(np.linalg.norm(robot_forward))
        facing_cosine = (
            float(np.dot(robot_forward / forward_norm, to_workspace / workspace_distance))
            if forward_norm > 1e-8 and workspace_distance > 1e-8
            else -1.0
        )
        geometry_pass = (
            MIN_FACING_COSINE <= facing_cosine
            and WORKSPACE_DISTANCE_RANGE_M[0] <= workspace_distance <= WORKSPACE_DISTANCE_RANGE_M[1]
            and workspace_diameter <= MAX_WORKSPACE_DIAMETER_M
        )
        if not geometry_pass:
            self._reject(
                "robot_not_facing_compact_island_workspace",
                {
                    "robot_base_pose": base_pose,
                    "workspace_center": workspace_center,
                    "workspace_distance_m": workspace_distance,
                    "workspace_diameter_m": workspace_diameter,
                    "facing_cosine": facing_cosine,
                    "thresholds": {
                        "min_facing_cosine": MIN_FACING_COSINE,
                        "workspace_distance_range_m": WORKSPACE_DISTANCE_RANGE_M,
                        "max_workspace_diameter_m": MAX_WORKSPACE_DIAMETER_M,
                    },
                },
            )

        self.accepted_initialization = {
            "scene_asset_provenance": "official MolmoSpaces iTHOR FloorPlan1",
            "placement_flow_provenance": (
                "official PackingTaskSampler/PickTaskSampler placement hooks with "
                "additive dual-pickup island filter and platform-edge alignment"
            ),
            "not_untouched_upstream_preset": True,
            "support": self._candidate_support,
            "official_placement_support_geom_id": self._candidate_support_geom_id,
            "support_top_z": support_top_z,
            "placement_aabb": placement_aabb,
            "max_abs_bottom_delta_m": max_abs_bottom_delta_m,
            "pickup_uid_to_name": uid_to_name,
            "box_name": box_name,
            "assignment": assignment,
            "base_frame_positions": base_positions,
            "robot_base_pose": base_pose,
            "platform_island_edge_alignment": edge_alignment,
            "workspace_center": workspace_center,
            "workspace_distance_m": workspace_distance,
            "workspace_diameter_m": workspace_diameter,
            "facing_cosine": facing_cosine,
            "geometry_pass": True,
            "same_support_pass": True,
            "robot_collision_pass": True,
            "reachability_policy": (
                "not evaluated during initialization; official sampler placement semantics "
                "are preserved and operator reachability is validated during teleoperation"
            ),
            "strict_pass": True,
        }


def build_tabletop_config(seed: int) -> BimanualYamPackingConfig:
    config = build_diagnostic_config(seed)
    config.task_sampler_config.task_sampler_class = FloorPlan1IslandDualPickupSampler
    return config


def sample_strict_tabletop_task(
    house_index: int,
    base_seed: int,
    max_attempts: int,
    *,
    output_dir: str | None = None,
    policy_dt_ms: int | None = None,
    task_horizon: int | None = None,
):
    """Boundedly resample clean official FloorPlan1 task instances.

    A failed sampler is always closed before the next attempt.  Only official
    retriable sampling failures are absorbed; programming/API errors propagate.
    The function never falls back to a non-island support.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    attempt_records: list[dict[str, Any]] = []
    for attempt_index in range(max_attempts):
        attempt_seed = base_seed + attempt_index
        config = build_tabletop_config(attempt_seed)
        if output_dir is not None:
            config.output_dir = output_dir
        if policy_dt_ms is not None:
            config.policy_dt_ms = policy_dt_ms
        if task_horizon is not None:
            config.task_horizon = task_horizon
        sampler = FloorPlan1IslandDualPickupSampler(config)
        try:
            task = sampler.sample_task(
                force_advance_scene=True,
                house_index=house_index,
            )
            accepted = sampler.accepted_initialization
            if task is None or not accepted or not accepted.get("strict_pass", False):
                raise HouseInvalidForTask(
                    "sample returned without a strict accepted tabletop initialization"
                )
            attempt_records.append(
                {
                    "attempt_index": attempt_index,
                    "attempt_seed": attempt_seed,
                    "status": "accepted",
                    "rejected": sampler.rejected_initializations,
                }
            )
            return sampler, task, attempt_records
        except (HouseInvalidForTask, RetriableError) as exc:
            attempt_records.append(
                {
                    "attempt_index": attempt_index,
                    "attempt_seed": attempt_seed,
                    "status": "rejected",
                    "error": f"{type(exc).__name__}: {exc}",
                    "rejected": sampler.rejected_initializations,
                }
            )
            sampler.close()

    error = HouseInvalidForTask(
        f"No strict FloorPlan1 island initialization passed in {max_attempts} attempts"
    )
    error.attempt_records = attempt_records
    raise error
