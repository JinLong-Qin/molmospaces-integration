from __future__ import annotations
import argparse, json, os
from pathlib import Path
import h5py, numpy as np

WORK = Path(os.environ.get("MOLMOSPACES_PNP_WORKDIR", "runtime/mimicgen_pick_and_place"))


def decode_json_row(row) -> dict:
    raw = bytes(row).rstrip(b"\0")
    return json.loads(raw or b"{}")


def pose_to_homogeneous(pose7: np.ndarray) -> np.ndarray:
    xyz = pose7[:3]
    w, x, y, z = pose7[3], pose7[4], pose7[5], pose7[6]
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm < 1e-9:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    else:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    H = np.eye(4, dtype=np.float32)
    H[:3, :3] = R
    H[:3, 3] = xyz.astype(np.float32)
    return H


def batched_pose_to_homogeneous(pose_batch: np.ndarray) -> np.ndarray:
    return np.stack([pose_to_homogeneous(p) for p in pose_batch], axis=0)


def gripper_qpos_from_rows(rows, n):
    out = np.zeros((n, 1), dtype=np.float32)
    last = 0.0
    for i in range(n):
        d = decode_json_row(rows[i])
        if "gripper" in d and len(d["gripper"]) >= 1:
            last = float(d["gripper"][0])
        out[i, 0] = last
    return out


def cumulative_signal_from_index(n: int, idx: int) -> np.ndarray:
    sig = np.zeros(n, dtype=np.uint8)
    if 0 <= idx < n:
        sig[idx:] = 1
    return sig


def first_index_where(arr, pred):
    for i, v in enumerate(arr):
        if pred(v):
            return i
    return -1


def build_demo(
    seed_index: int,
    demo_name: str,
    out_data,
    manifest,
    replay_root: Path,
    manual_review_exceptions: set[int],
    action_type: str,
):
    seed = manifest["seeds"][seed_index]
    raw_dir = seed.get("raw_h5_dir", "raw")
    raw_dir = Path(raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = WORK / "artifacts/seeds" / raw_dir
    raw_h5 = raw_dir / seed["raw_h5"]
    traj = seed["traj_key"]
    replay_dir = replay_root / f"seed_{seed_index:02d}"
    result_path = replay_dir / "datagen_info_collection_result.json"
    if not result_path.exists():
        raise RuntimeError(f"seed_{seed_index:02d}: missing datagen result {result_path}")
    replay_result = json.loads(result_path.read_text())
    automatic_hard_pass = bool(
        replay_result.get("final_success") and replay_result.get("success_persistent_to_end")
    )
    manually_approved = bool(
        seed_index in manual_review_exceptions and replay_result.get("final_success")
    )
    if not (automatic_hard_pass or manually_approved):
        raise RuntimeError(f"seed_{seed_index:02d} is not hard-pass: {replay_result}")
    pickup_rel_obs = np.load(replay_dir / "pickup_obj_pose_rel_observations.npy")
    place_rel_obs = np.load(replay_dir / "place_receptacle_pose_rel_observations.npy")
    target_pose_all = np.load(replay_dir / "target_pose_rel_actions.npy")
    with h5py.File(raw_h5, "r") as f:
        g = f[traj]
        obs_scene = json.loads(bytes(g["obs_scene"][()]).rstrip(b"\0"))
        commanded_all = [decode_json_row(x) for x in g["actions/commanded_action"][:]]
        tcp_delta_all = [decode_json_row(x) for x in g["actions/ee_twist"][:]]
        real_action_rows = [
            i
            for i, d in enumerate(commanded_all)
            if 0 < i < len(commanded_all) - 1
            and "arm" in d
            and "gripper" in d
            and len(d.get("arm", [])) == 7
            and len(d.get("gripper", [])) >= 1
            and len(tcp_delta_all[i].get("arm", [])) == 6
        ]
        if not real_action_rows or real_action_rows[0] != 1:
            raise RuntimeError(
                f"seed_{seed_index:02d}: unexpected real action rows start {real_action_rows[:3]}"
            )
        n = len(real_action_rows)
        joint_position_actions = np.zeros((n, 8), dtype=np.float32)
        for j, row_i in enumerate(real_action_rows):
            d = commanded_all[row_i]
            joint_position_actions[j, :7] = np.asarray(d["arm"], dtype=np.float32)
            joint_position_actions[j, 7] = float(d["gripper"][0])
        tcp_delta_actions = np.zeros((n, 7), dtype=np.float32)
        for j, row_i in enumerate(real_action_rows):
            d = tcp_delta_all[row_i]
            arm = np.asarray(d.get("arm", []), dtype=np.float32)
            if arm.shape != (6,):
                raise RuntimeError(
                    f"seed_{seed_index:02d}: invalid actions/ee_twist arm at row {row_i}: "
                    f"shape={arm.shape}"
                )
            tcp_delta_actions[j, :6] = arm
            tcp_delta_actions[j, 6] = joint_position_actions[j, 7]
        actions = tcp_delta_actions if action_type == "tcp_delta" else joint_position_actions
        pre_obs = np.asarray(real_action_rows, dtype=int) - 1
        post_obs = np.asarray(real_action_rows, dtype=int)
        if post_obs.max() >= len(g["success"]):
            raise RuntimeError(f"seed_{seed_index:02d}: post obs out of bounds")
        tcp7 = g["obs/extra/tcp_pose"][pre_obs].astype(np.float32)
        eef_hmat = batched_pose_to_homogeneous(tcp7)
        states = g["env_states/articulations/panda"][pre_obs].astype(np.float32)
        rewards = g["rewards"][post_obs].astype(np.float32)
        success = g["success"][post_obs].astype(bool)
        phases_pre = g["obs/extra/policy_phase"][pre_obs].astype(np.int64)
        phases_post = g["obs/extra/policy_phase"][post_obs].astype(np.int64)
        gripper_qpos = gripper_qpos_from_rows(g["obs/agent/qpos"], len(g["obs/agent/qpos"]))[
            pre_obs
        ]
    pickup_obj_pose = pickup_rel_obs[pre_obs].astype(np.float32)
    place_receptacle_pose = place_rel_obs[pre_obs].astype(np.float32)
    target_pose = target_pose_all[:n].astype(np.float32)
    if target_pose.shape != (n, 4, 4):
        raise RuntimeError(f"seed_{seed_index:02d}: target_pose shape {target_pose.shape}, n={n}")
    idx_pregrasp_done = first_index_where(phases_post, lambda p: int(p) >= 3)
    idx_grasp_done = first_index_where(phases_post, lambda p: int(p) >= 4)
    idx_gripper_closed = first_index_where(phases_post, lambda p: int(p) >= 5)
    idx_lift_done = first_index_where(phases_post, lambda p: int(p) >= 6)
    idx_preplace_done = first_index_where(phases_post, lambda p: int(p) >= 7)
    idx_place_success = int(np.argmax(success)) if success.any() else -1
    signals = {
        "pregrasp_done": cumulative_signal_from_index(n, idx_pregrasp_done),
        "grasp_done": cumulative_signal_from_index(n, idx_grasp_done),
        "gripper_closed": cumulative_signal_from_index(n, idx_gripper_closed),
        "lift_done": cumulative_signal_from_index(n, idx_lift_done),
        "preplace_done": cumulative_signal_from_index(n, idx_preplace_done),
        "place_success": cumulative_signal_from_index(n, idx_place_success),
    }
    signal_indices = {k: int(np.argmax(v)) if v.any() else -1 for k, v in signals.items()}
    ordered = [
        signal_indices[k]
        for k in [
            "pregrasp_done",
            "grasp_done",
            "gripper_closed",
            "lift_done",
            "preplace_done",
            "place_success",
        ]
    ]
    if any(v < 0 for v in ordered) or ordered != sorted(ordered):
        raise RuntimeError(f"seed_{seed_index:02d}: invalid signals {signal_indices}")
    dones = np.zeros(n, dtype=np.int32)
    if success.any():
        dones[int(np.argmax(success)) :] = 1
    else:
        dones[-1] = 1
    demo = out_data.create_group(demo_name)
    demo.create_dataset("actions", data=actions, compression="gzip")
    demo.create_dataset(
        "source_joint_position_actions", data=joint_position_actions, compression="gzip"
    )
    demo.create_dataset("source_tcp_delta_actions", data=tcp_delta_actions, compression="gzip")
    demo.create_dataset("states", data=states, compression="gzip")
    demo.create_dataset("dones", data=dones)
    demo.create_dataset("rewards", data=rewards)
    obs = demo.create_group("obs")
    obs.create_dataset("robot0_eef_pos", data=tcp7[:, :3])
    obs.create_dataset("robot0_eef_quat", data=tcp7[:, 3:7])
    obs.create_dataset("robot0_gripper_qpos", data=gripper_qpos)
    obs.create_dataset("object_pos", data=pickup_obj_pose[:, :3, 3])
    obs.create_dataset("place_receptacle_pos", data=place_receptacle_pose[:, :3, 3])
    obs.create_dataset("tcp_pos", data=tcp7[:, :3])
    obs.create_dataset("tcp_quat", data=tcp7[:, 3:7])
    obs.create_dataset("policy_phase", data=phases_pre)
    dgi = demo.create_group("datagen_info")
    dgi.attrs["env_interface_name"] = "MG_MolmoSpacesPickAndPlace"
    dgi.attrs["env_interface_type"] = "molmospaces"
    dgi.create_dataset("eef_pose", data=eef_hmat, compression="gzip")
    dgi.create_dataset("target_pose", data=target_pose, compression="gzip")
    dgi.create_dataset("gripper_action", data=joint_position_actions[:, 7:8].astype(np.float32))
    op = dgi.create_group("object_poses")
    op.create_dataset("pickup_obj", data=pickup_obj_pose, compression="gzip")
    op.create_dataset("place_receptacle", data=place_receptacle_pose, compression="gzip")
    sig = dgi.create_group("subtask_term_signals")
    for k, v in signals.items():
        sig.create_dataset(k, data=v)
    demo.attrs["source_seed_index"] = int(seed_index)
    demo.attrs["house_id"] = int(seed["house_id"])
    demo.attrs["batch_id"] = int(seed["batch_id"])
    demo.attrs["traj_index"] = int(seed["traj_index"])
    demo.attrs["source_h5"] = seed["raw_h5"]
    demo.attrs["source_h5_file"] = seed.get("source_h5_file", str(raw_h5))
    demo.attrs["source_run_root"] = seed.get("source_run_root", "")
    demo.attrs["num_samples"] = n
    demo.attrs["source_observations"] = int(seed["length"])
    demo.attrs["alignment"] = (
        "raw first dummy and final done-sentinel rows excluded; pre_obs=row-1, post_obs=row"
    )
    demo.attrs["action_type"] = action_type
    demo.attrs["action_semantics"] = (
        "6D body-frame commanded EEF twist delta [dx,dy,dz,dRx,dRy,dRz] + gripper"
        if action_type == "tcp_delta"
        else "7D absolute arm joint-position target + gripper"
    )
    demo.attrs["automatic_hard_pass"] = automatic_hard_pass
    demo.attrs["manual_review_exception"] = manually_approved and not automatic_hard_pass
    demo.attrs["seed_kind"] = "synthetic_planner_expert"
    demo.attrs["task_type"] = "pick_and_place"
    demo.attrs["task_description"] = obs_scene.get("task_description", "")
    demo.attrs["pickup_obj_name"] = obs_scene.get("object_name")
    demo.attrs["place_receptacle_name"] = obs_scene.get("place_receptacle_name")
    demo.attrs["subtask_signal_indices"] = json.dumps(signal_indices)
    return {
        "demo": demo_name,
        "seed_index": seed_index,
        "house_id": int(seed["house_id"]),
        "n": n,
        "action_type": action_type,
        "signals": signal_indices,
        "first_success": int(np.argmax(success)) if success.any() else -1,
        "success_final": bool(success[-1]),
        "task": obs_scene.get("task_description", ""),
    }


def parse_indices(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", default="all")
    ap.add_argument(
        "--action-type",
        choices=("joint_position", "tcp_delta"),
        default="joint_position",
        help="Training action stored in data/demo_*/actions; both representations remain for audit.",
    )
    ap.add_argument(
        "--manual-review-exceptions",
        default="",
        help="Comma-separated final-success seed indices approved by manual video review.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=WORK / "artifacts/seeds/pnp_source_manifest.json",
    )
    ap.add_argument("--replay-root", type=Path, default=WORK / "artifacts/replay_source_candidates")
    ap.add_argument(
        "--out",
        type=Path,
        default=WORK / "artifacts/seeds/robomimic_pnp_source.hdf5",
    )
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    accepted = (
        list(range(len(manifest["seeds"])))
        if args.accepted.strip().lower() == "all"
        else parse_indices(args.accepted)
    )
    if not accepted:
        raise RuntimeError("--accepted resolved to no source trajectories")
    if min(accepted) < 0 or max(accepted) >= len(manifest["seeds"]):
        raise RuntimeError(
            f"accepted index out of manifest range: max={max(accepted)} n={len(manifest['seeds'])}"
        )
    manual_review_exceptions = (
        set()
        if not args.manual_review_exceptions.strip()
        else set(parse_indices(args.manual_review_exceptions))
    )
    if not manual_review_exceptions.issubset(accepted):
        raise RuntimeError("--manual-review-exceptions must be included in --accepted")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    summaries = []
    total = 0
    with h5py.File(args.out, "w") as out:
        data = out.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": "MolmoSpacesPickAndPlaceEnv",
                "env_kwargs": {
                    "config": "FrankaPickAndPlaceOmniCamConfig",
                    "source_demo_count": len(accepted),
                    "generation_strategy": "cross-demo subtask mixing via --select-src-per-subtask",
                    "action_type": args.action_type,
                },
                "type": 2,
                "note": "MolmoSpaces PnP synthetic planner expert source demos for MimicGen cross-demo subtask-mixing integration",
            }
        )
        for demo_i, seed_index in enumerate(accepted):
            s = build_demo(
                seed_index,
                f"demo_{demo_i}",
                data,
                manifest,
                args.replay_root,
                manual_review_exceptions,
                args.action_type,
            )
            summaries.append(s)
            total += s["n"]
        data.attrs["total"] = int(total)
        out.attrs["provenance"] = (
            "MolmoBot-Data allenai/molmobot-data / FrankaPickAndPlaceOmniCamConfig"
        )
        out.attrs["seed_kind"] = (
            "synthetic_planner_expert (cuRobo/scripted). NOT human demonstration."
        )
        out.attrs["n_demos"] = len(accepted)
        out.attrs["action_type"] = args.action_type
        out.attrs["accepted_seed_indices"] = json.dumps(accepted)
        out.attrs["manual_review_exception_seed_indices"] = json.dumps(
            sorted(manual_review_exceptions)
        )
        out.attrs["source_run_roots"] = json.dumps(
            sorted({str(manifest["seeds"][i].get("source_run_root", "")) for i in accepted})
        )
        out.attrs["generation_intent"] = (
            "source pool for MimicGen cross-source/subtask recombination; not single-source whole-trajectory baseline"
        )
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "out": str(args.out),
                "manifest": str(args.manifest),
                "replay_root": str(args.replay_root),
                "accepted": accepted,
                "manual_review_exceptions": sorted(manual_review_exceptions),
                "action_type": args.action_type,
                "n_demos": len(accepted),
                "total": total,
                "demos": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(json.dumps(json.loads(summary_path.read_text()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
