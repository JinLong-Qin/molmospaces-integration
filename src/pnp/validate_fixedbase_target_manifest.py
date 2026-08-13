from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reset-only fixed-base PnP targets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--house-id", type=int, required=True)
    parser.add_argument(
        "--robot-base-pose",
        type=float,
        nargs=7,
        required=True,
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        help="Fixed robot base position and quaternion for this scenario.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.input.read_text())
    rows = manifest.get("seeds", [])
    errors: list[str] = []
    layout_hashes: set[str] = set()
    static_hashes: set[str] = set()
    camera_hashes: set[str] = set()

    if manifest.get("purpose") != "MimicGen target resets; no planner policy or task actions executed":
        errors.append("manifest purpose does not identify reset-only sampling")
    if len(rows) != args.expected_count or manifest.get("n_targets") != args.expected_count:
        errors.append(
            f"target count mismatch: rows={len(rows)} metadata={manifest.get('n_targets')} "
            f"expected={args.expected_count}"
        )

    for i, row in enumerate(rows):
        spec = row.get("episode_spec")
        if not isinstance(spec, dict):
            errors.append(f"row {i}: missing episode_spec")
            continue
        task = spec.get("task", {})
        if row.get("target_index") != i:
            errors.append(f"row {i}: target_index mismatch")
        if row.get("target_kind") != "reset_only_sampled_episode_spec":
            errors.append(f"row {i}: invalid target_kind")
        if spec.get("house_index") != args.house_id or row.get("house_id") != args.house_id:
            errors.append(f"row {i}: house mismatch")
        if not np.allclose(task.get("robot_base_pose", []), args.robot_base_pose, atol=1e-6):
            errors.append(f"row {i}: robot base mismatch")
        if not np.isinf(task.get("succ_pos_threshold", np.nan)):
            errors.append(f"row {i}: succ_pos_threshold must be infinity for PickAndPlace")
        pickup_pose = task.get("pickup_obj_start_pose")
        place_pose = task.get("place_receptacle_start_pose")
        if not (isinstance(pickup_pose, list) and len(pickup_pose) == 7):
            errors.append(f"row {i}: invalid pickup pose")
            continue
        if not (isinstance(place_pose, list) and len(place_pose) == 7):
            errors.append(f"row {i}: invalid place pose")
            continue
        if not np.isfinite(np.asarray(pickup_pose + place_pose, dtype=float)).all():
            errors.append(f"row {i}: non-finite object pose")
        layout_hash = digest(
            {"pickup_pose": pickup_pose, "place_receptacle_pose": place_pose}
        )
        if layout_hash != row.get("layout_sha256"):
            errors.append(f"row {i}: layout hash mismatch")
        layout_hashes.add(layout_hash)
        camera_hashes.add(digest(spec.get("cameras", [])))
        object_poses = spec.get("scene_modifications", {}).get("object_poses", {})
        target_names = {task.get("pickup_obj_name"), task.get("place_receptacle_name")}
        non_target_object_poses = {
            name: pose for name, pose in object_poses.items() if name not in target_names
        }
        static_hashes.add(
            digest(
                {
                    "house_index": spec.get("house_index"),
                    "scene_dataset": spec.get("scene_dataset"),
                    "data_split": spec.get("data_split"),
                    # The sampler may choose a valid Franka arm reset qpos per
                    # target. That is target reset state, not static scene.
                    "robot_name": spec.get("robot", {}).get("robot_name"),
                    "img_resolution": spec.get("img_resolution"),
                    "added_objects": spec.get("scene_modifications", {}).get("added_objects", {}),
                    "removed_objects": spec.get("scene_modifications", {}).get("removed_objects", []),
                    "non_target_object_poses": non_target_object_poses,
                    "pickup_obj_name": task.get("pickup_obj_name"),
                    "place_receptacle_name": task.get("place_receptacle_name"),
                    "robot_base_pose": task.get("robot_base_pose"),
                }
            )
        )

    if len(layout_hashes) != args.expected_count:
        errors.append(f"unique layouts={len(layout_hashes)}, expected={args.expected_count}")
    if len(static_hashes) != 1:
        errors.append(f"static fingerprints={len(static_hashes)}, expected=1")
    if len(camera_hashes) != 1:
        errors.append(f"camera fingerprints={len(camera_hashes)}, expected=1")

    report = {
        "valid": not errors,
        "targets": len(rows),
        "unique_layouts": len(layout_hashes),
        "static_fingerprints": len(static_hashes),
        "camera_fingerprints": len(camera_hashes),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
