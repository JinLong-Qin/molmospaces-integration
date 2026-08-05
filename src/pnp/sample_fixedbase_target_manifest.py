from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("MOLMOSPACES_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT))


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample fixed-base MolmoSpaces PnP target EpisodeSpecs without running a policy."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--house-id", type=int, default=1716)
    parser.add_argument(
        "--pickup-object",
        default="Irishpotato_4ccdc5ebde4d6fee07ff9eefb0b60cfb_1_0_2",
    )
    parser.add_argument(
        "--place-receptacle-uid", default="5c5c3b9ae7874b709c10ac57dad33195"
    )
    parser.add_argument(
        "--robot-base-pose",
        type=float,
        nargs=7,
        default=[0.813989, 14.103546, 0.329206, -0.631043, 0.0, 0.0, 0.775748],
    )
    parser.add_argument("--pickup-min-dist", type=float, default=0.02)
    parser.add_argument("--pickup-max-dist", type=float, default=0.12)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="Existing validated source manifest; used only for EpisodeSpec source provenance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.max_attempts < args.count:
        raise ValueError("count must be positive and max-attempts must be at least count")

    source_manifest = json.loads(args.source_manifest.read_text())
    source = source_manifest["seeds"][0]
    source_h5 = Path(source.get("source_h5_file") or source["raw_h5_dir"])
    if source_h5.is_dir():
        source_h5 /= source["raw_h5"]
    source_traj = source["traj_key"]
    source_length = int(source.get("length", 0)) or None

    # Avoid runtime network downloads while importing MolmoSpaces metadata helpers.
    try:
        import nltk

        original_download = nltk.download
        nltk.download = lambda *a, **k: True
        try:
            import molmo_spaces.utils.synset_utils  # noqa: F401
        finally:
            nltk.download = original_download
    except ModuleNotFoundError:
        pass

    from molmo_spaces.configs.camera_configs import FrankaDroidCameraSystem
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaPickAndPlaceDroidDataGenConfig,
    )
    from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
    from scripts.benchmarks.create_json_benchmark import (
        extract_frozen_config,
        frozen_config_to_episode_spec,
    )

    cfg = FrankaPickAndPlaceDroidDataGenConfig()
    cfg.seed = args.seed
    cfg.scene_dataset = "procthor-objaverse"
    cfg.data_split = "val"
    cfg.robot_config = FrankaRobotConfig()
    cfg.camera_config = FrankaDroidCameraSystem()
    cfg.task_sampler_config.house_inds = [args.house_id]
    cfg.task_sampler_config.samples_per_house = args.max_attempts
    cfg.task_sampler_config.fixed_pickup_obj_name = args.pickup_object
    cfg.task_sampler_config.randomize_fixed_pickup_pose = True
    cfg.task_sampler_config.fixed_pickup_placement_radius_range = (
        args.pickup_min_dist,
        args.pickup_max_dist,
    )
    cfg.task_sampler_config.fixed_place_receptacle_uid = args.place_receptacle_uid
    cfg.task_sampler_config.num_place_receptacles = 1
    cfg.task_sampler_config.episodes_per_receptacle = 0
    cfg.task_sampler_config.fixed_robot_base_pose = list(args.robot_base_pose)
    cfg.task_sampler_config.randomize_lighting = False
    cfg.task_sampler_config.randomize_textures = False
    cfg.task_sampler_config.randomize_dynamics = False
    cfg.task_config.pickup_obj_name = args.pickup_object
    cfg.profile = False

    sampler = PickAndPlaceTaskSampler(cfg)
    sampler.seed_task_sampling(args.seed)
    entries: list[dict] = []
    layout_hashes: set[str] = set()
    rejected = 0

    try:
        for attempt in range(args.max_attempts):
            if len(entries) >= args.count:
                break
            try:
                task = sampler.sample_task(house_index=args.house_id)
                if task is None:
                    raise RuntimeError("sampler exhausted before target count was reached")
                task.reset()
                obs_scene = task.get_obs_scene()
                frozen = extract_frozen_config(obs_scene)
                spec = frozen_config_to_episode_spec(
                    frozen_config=frozen,
                    obs_scene=obs_scene,
                    house_id=args.house_id,
                    scene_dataset="procthor-objaverse",
                    data_split="val",
                    source_h5_file=str(source_h5),
                    source_traj_key=source_traj,
                    source_episode_length=source_length,
                    img_resolution=tuple(cfg.camera_config.img_resolution),
                    camera_system_class=type(cfg.camera_config).__name__,
                    task_horizon_sec=30,
                )
                spec.task["task_cls"] = spec.task["task_cls"].replace(
                    "mujoco_thor.", "molmo_spaces.", 1
                )
                spec.task["task_type"] = "pick_and_place"
                spec.task["max_place_receptacle_pos_displacement"] = 0.15
                spec.task["max_place_receptacle_rot_displacement"] = float(np.radians(60))
                spec_dict = spec.model_dump(mode="json")
                # Pydantic's JSON mode maps infinity to null by default, but
                # PickAndPlaceTaskSpec requires a float and uses infinity to
                # disable the lift-distance success check.
                spec_dict["task"]["succ_pos_threshold"] = float("inf")
                pickup_pose = spec.task["pickup_obj_start_pose"]
                place_pose = spec.task["place_receptacle_start_pose"]
                layout_hash = canonical_hash(
                    {"pickup_pose": pickup_pose, "place_receptacle_pose": place_pose}
                )
                if layout_hash in layout_hashes:
                    rejected += 1
                    continue
                layout_hashes.add(layout_hash)
                entries.append(
                    {
                        "target_index": len(entries),
                        "sampling_attempt": attempt,
                        "house_id": args.house_id,
                        "layout_sha256": layout_hash,
                        "pickup_obj_name": spec.task["pickup_obj_name"],
                        "place_receptacle_name": spec.task["place_receptacle_name"],
                        "pickup_pose": pickup_pose,
                        "place_receptacle_pose": place_pose,
                        "robot_base_pose": spec.task["robot_base_pose"],
                        "episode_spec": spec_dict,
                        "target_kind": "reset_only_sampled_episode_spec",
                    }
                )
                print(
                    f"accepted {len(entries):04d}/{args.count} attempt={attempt:04d} "
                    f"layout={layout_hash[:12]}",
                    flush=True,
                )
            except Exception as exc:
                rejected += 1
                print(f"rejected attempt={attempt:04d}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        sampler.close()

    if len(entries) != args.count:
        raise RuntimeError(
            f"sampled only {len(entries)}/{args.count} unique targets after {args.max_attempts} attempts"
        )

    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "MimicGen target resets; no planner policy or task actions executed",
        "source_demo_pool": str(args.source_manifest),
        "sampler": {
            "seed": args.seed,
            "house_id": args.house_id,
            "pickup_object": args.pickup_object,
            "place_receptacle_uid": args.place_receptacle_uid,
            "robot_base_pose": list(args.robot_base_pose),
            "pickup_radius_range": [args.pickup_min_dist, args.pickup_max_dist],
            "randomize_scene": False,
        },
        "n_targets": len(entries),
        "n_rejected_attempts": rejected,
        "seeds": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {len(entries)} reset-only targets to {args.output}")


if __name__ == "__main__":
    main()
