from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path

import h5py
import numpy as np
import zstandard as zstd

WORK = Path(os.environ.get("MOLMOSPACES_PNP_WORKDIR", "runtime/mimicgen_pick_and_place"))
DEFAULT_TAR = None
OUT = WORK / "artifacts/seeds"
RAW = OUT / "raw_source_pool"

REQUIRED_FRANKA_PATHS = (
    "actions/commanded_action",
    "env_states/articulations/panda",
    "obs/extra/obj_start",
    "obs/extra/robot_base_pose",
    "obs/extra/policy_phase",
    "obs/extra/task_info",
    "obs_scene",
    "success",
)
EXPECTED_PNP_PHASES = set(range(10))


def decode_json_blob(value):
    try:
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        return json.loads(bytes(value).rstrip(b"\0").decode("utf-8"))
    except Exception:
        return {}


def action_fingerprint(dataset) -> str:
    digest = hashlib.sha256()
    values = np.asarray(dataset)
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def initial_state_fingerprint(group, obs_scene: dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(obs_scene, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for path in (
        "obs/extra/obj_start",
        "obs/extra/robot_base_pose",
        "env_states/articulations/panda",
    ):
        if path in group:
            values = np.asarray(group[path][0])
            digest.update(path.encode("utf-8"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
    return digest.hexdigest()


def scan_h5(h5, *, strict_franka: bool, require_persistent_success: bool = True):
    out = []
    mask = np.asarray(h5.get("valid_traj_mask", []), dtype=bool)
    keys = sorted(
        [x for x in h5.keys() if x.startswith("traj_")],
        key=lambda s: int(s.split("_")[1]),
    )
    for key in keys:
        traj_index = int(key.split("_")[1])
        if len(mask) and traj_index < len(mask) and not bool(mask[traj_index]):
            continue
        group = h5[key]
        if "success" not in group:
            continue
        success = np.asarray(group["success"][:], dtype=bool)
        if not (len(success) and bool(success[-1])):
            continue
        first_success = int(np.argmax(success)) if success.any() else -1
        persistent = first_success >= 0 and bool(success[first_success:].all())
        if strict_franka and require_persistent_success and not persistent:
            continue

        obs_scene = decode_json_blob(group["obs_scene"][()]) if "obs_scene" in group else {}
        if obs_scene.get("task_type") != "pick_and_place":
            continue

        missing_paths = [path for path in REQUIRED_FRANKA_PATHS if path not in group]
        phases = set()
        if "obs/extra/policy_phase" in group:
            phases = {int(x) for x in np.asarray(group["obs/extra/policy_phase"][:]).reshape(-1)}
        if strict_franka and (missing_paths or not EXPECTED_PNP_PHASES.issubset(phases)):
            continue

        task_info = {}
        try:
            task_info = decode_json_blob(group["obs/extra/task_info"][-1])
        except Exception:
            pass
        if strict_franka and task_info.get("success") is not True:
            continue

        action_sha256 = None
        if "actions/commanded_action" in group:
            action_sha256 = action_fingerprint(group["actions/commanded_action"])
        state_sha256 = initial_state_fingerprint(group, obs_scene)
        combined_sha256 = hashlib.sha256(
            f"{state_sha256}:{action_sha256}".encode("ascii")
        ).hexdigest()
        out.append(
            {
                "traj_index": traj_index,
                "traj_key": key,
                "length": int(len(success)),
                "reward_sum": float(np.asarray(group["rewards"][:]).sum())
                if "rewards" in group
                else 0.0,
                "success_final": bool(success[-1]),
                "success_persistent_from_first": persistent,
                "first_success": first_success,
                "object_name": obs_scene.get("object_name"),
                "pickup_obj_name": obs_scene.get("pickup_obj_name"),
                "place_receptacle_name": obs_scene.get("place_receptacle_name"),
                "task_type": obs_scene.get("task_type"),
                "task_description": obs_scene.get("task_description"),
                "final_task_info": task_info,
                "obs_scene_keys": sorted(obs_scene.keys())[:40],
                "policy_phases": sorted(phases),
                "initial_task_state_sha256": state_sha256,
                "action_trajectory_sha256": action_sha256,
                "trajectory_fingerprint_sha256": combined_sha256,
            }
        )
    return out


def scan_h5_bytes(data: bytes):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as handle:
        handle.write(data)
        tmp = handle.name
    try:
        with h5py.File(tmp, "r") as h5:
            return scan_h5(h5, strict_franka=False)
    finally:
        os.unlink(tmp)


def select_from_molmobot_shard(tar_path: Path, count: int):
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_raw = set()
    with tarfile.open(tar_path, "r") as outer:
        members = [
            member
            for member in outer.getmembers()
            if member.isfile() and member.name.endswith(".tar.zst") and "house_" in member.name
        ]
        for member in members:
            if len(rows) >= count:
                break
            match = re.search(r"house_(\d+)", member.name)
            house_id = int(match.group(1)) if match else -1
            print("HOUSE", house_id, flush=True)
            compressed = outer.extractfile(member).read()
            reader = zstd.ZstdDecompressor().stream_reader(io.BytesIO(compressed))
            try:
                with tarfile.open(fileobj=reader, mode="r|") as inner:
                    for h5_member in inner:
                        if len(rows) >= count:
                            break
                        if not h5_member.isfile() or not h5_member.name.endswith(".h5"):
                            continue
                        data = inner.extractfile(h5_member).read()
                        candidates = scan_h5_bytes(data)
                        if not candidates:
                            continue
                        batch_match = re.search(r"trajectories_batch_(\d+)_", h5_member.name)
                        batch_id = int(batch_match.group(1)) if batch_match else -1
                        raw_name = f"source_pool_house{house_id}__batch{batch_id}.h5"
                        if raw_name not in seen_raw:
                            (RAW / raw_name).write_bytes(data)
                            seen_raw.add(raw_name)
                        for candidate in candidates:
                            if len(rows) >= count:
                                break
                            candidate.update(
                                {
                                    "seed_index": len(rows),
                                    "house_id": house_id,
                                    "batch_id": batch_id,
                                    "source_outer_member": member.name,
                                    "source_inner_h5": h5_member.name,
                                    "raw_h5_dir": "raw_source_pool",
                                    "raw_h5": raw_name,
                                }
                            )
                            rows.append(candidate)
                            print(
                                "SELECT", len(rows) - 1, house_id, candidate["traj_key"], flush=True
                            )
            finally:
                reader.close()
    return {
        "dataset": "FrankaPickAndPlaceOmniCamConfig",
        "split": "val_shards/00000.tar",
        "input_kind": "molmobot_shard",
        "selection_note": (
            f"first {count} successful Pick-and-Place source demos from official shard for "
            "MimicGen cross-demo subtask-mixing source pool; synthetic planner expert, not human demos"
        ),
        "seeds": rows,
    }


def infer_run_root(h5_path: Path, supplied_root: Path) -> Path:
    parent = h5_path.parent
    if re.fullmatch(r"house_\d+", parent.name):
        return parent.parent
    return supplied_root


def select_from_franka_datagen(
    root: Path,
    count: int,
    house_id_filter: int | None = None,
    run_name_prefix: str | None = None,
    require_persistent_success: bool = True,
):
    h5_paths = sorted(root.rglob("trajectories_batch_*.h5"))
    if not h5_paths:
        raise SystemExit(f"no trajectories_batch_*.h5 files found under {root}")

    rows = []
    seen_fingerprints = set()
    duplicate_count = 0
    for h5_path in h5_paths:
        if len(rows) >= count:
            break
        house_match = re.search(r"house_(\d+)", str(h5_path))
        batch_match = re.search(r"trajectories_batch_(\d+)_", h5_path.name)
        house_id = int(house_match.group(1)) if house_match else -1
        batch_id = int(batch_match.group(1)) if batch_match else -1
        if house_id_filter is not None and house_id != house_id_filter:
            continue
        run_root = infer_run_root(h5_path, root)
        if run_name_prefix is not None and not run_root.name.startswith(run_name_prefix):
            continue
        with h5py.File(h5_path, "r") as h5:
            candidates = scan_h5(
                h5,
                strict_franka=True,
                require_persistent_success=require_persistent_success,
            )
        for candidate in candidates:
            if len(rows) >= count:
                break
            fingerprint = candidate["trajectory_fingerprint_sha256"]
            if not fingerprint or fingerprint in seen_fingerprints:
                duplicate_count += 1
                continue
            seen_fingerprints.add(fingerprint)
            candidate.update(
                {
                    "seed_index": len(rows),
                    "house_id": house_id,
                    "batch_id": batch_id,
                    "source_run_root": str(run_root.resolve()),
                    "source_h5_file": str(h5_path.resolve()),
                    "raw_h5_dir": str(h5_path.parent.resolve()),
                    "raw_h5": h5_path.name,
                    "seed_kind": "synthetic_scripted_ik_planner_expert",
                }
            )
            rows.append(candidate)
            print(
                "SELECT",
                len(rows) - 1,
                house_id,
                candidate["traj_key"],
                fingerprint[:12],
                flush=True,
            )

    if len(rows) < count:
        raise SystemExit(
            f"only found {len(rows)}/{count} unique strict-success Franka PnP trajectories "
            f"under {root} (skipped duplicates={duplicate_count})"
        )
    return {
        "dataset": "FrankaPickAndPlaceDroidDataGenConfig",
        "split": None,
        "input_kind": "franka_datagen_hdf5",
        "franka_datagen_root": str(root.resolve()),
        "house_id_filter": house_id_filter,
        "run_name_prefix": run_name_prefix,
        "raw_success_requirement": "terminal_and_persistent"
        if require_persistent_success
        else "terminal_only; replay hard-pass required before conversion",
        "selection_note": (
            f"{count} unique strict-success Franka Pick-and-Place trajectories selected from "
            "locally generated MolmoSpaces HDF5; synthetic scripted-IK planner expert demos, "
            "not human demonstrations and not RB-Y1 planner-server trajectories"
        ),
        "deduplication": (
            "sha256 over complete scene metadata, initial object/robot state, and raw actions/commanded_action bytes"
        ),
        "skipped_duplicate_trajectories": duplicate_count,
        "seeds": rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Select a MimicGen PnP source pool from the original MolmoBot shard or Franka datagen HDF5 files."
    )
    parser.add_argument(
        "--source-shard",
        type=Path,
        help="Explicit official MolmoBot .tar shard to scan.",
    )
    parser.add_argument(
        "--franka-datagen-root",
        type=Path,
        help="Read house_*/trajectories_batch_*.h5 from this Franka datagen run/collection instead of the MolmoBot shard.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=int(os.environ.get("PNP_SOURCE_COUNT", "17")),
        help="Number of unique strict-success source trajectories to select.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT / "pnp_source_manifest.json",
        help="Destination manifest path.",
    )
    parser.add_argument(
        "--house-id",
        type=int,
        help="Restrict Franka datagen selection to one house ID.",
    )
    parser.add_argument(
        "--run-name-prefix",
        help="Restrict Franka datagen selection to run directory names with this prefix.",
    )
    parser.add_argument(
        "--allow-nonpersistent-candidates",
        action="store_true",
        help="Include terminal-only candidates for replay audit; conversion still requires replay hard-pass.",
    )
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit(f"--count must be positive, got {args.count}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.source_shard is not None and args.franka_datagen_root is not None:
        raise SystemExit("choose exactly one of --source-shard or --franka-datagen-root")
    if args.source_shard is not None:
        shard = args.source_shard.expanduser().resolve()
        if not shard.is_file():
            raise SystemExit(f"source shard is not a file: {shard}")
        manifest = select_from_molmobot_shard(shard, args.count)
    elif args.franka_datagen_root is None:
        raise SystemExit("one of --source-shard or --franka-datagen-root is required")
    else:
        root = args.franka_datagen_root.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"Franka datagen root is not a directory: {root}")
        manifest = select_from_franka_datagen(
            root,
            args.count,
            args.house_id,
            args.run_name_prefix,
            not args.allow_nonpersistent_candidates,
        )

    if len(manifest["seeds"]) < args.count:
        raise SystemExit(f"only found {len(manifest['seeds'])}/{args.count}")
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(args.out),
                "n": len(manifest["seeds"]),
                "input_kind": manifest["input_kind"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
