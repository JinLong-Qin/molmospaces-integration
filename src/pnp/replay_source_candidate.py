from __future__ import annotations
import argparse, json, os, signal, sys, time
from pathlib import Path

signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("hard timeout")))
signal.alarm(1800)
ROOT = Path(os.environ.get("MOLMOSPACES_ROOT", "."))
sys.path.insert(0, str(ROOT))
WORK = Path(
    os.environ.get("MOLMOSPACES_PNP_WORKDIR", str(ROOT / "runtime/mimicgen_pick_and_place"))
)
ap = argparse.ArgumentParser()
ap.add_argument("--seed-index", type=int, required=True)
ap.add_argument(
    "--manifest",
    default=str(WORK / "artifacts/seeds/pnp_source_manifest.json"),
)
ap.add_argument("--out-root", default=str(WORK / "artifacts/replay_source_candidates"))
args = ap.parse_args()
T0 = time.monotonic()


def log(s):
    print(f"[{time.monotonic() - T0:7.1f}s] {s}", flush=True)


manifest_path = Path(args.manifest)
manifest = json.loads(manifest_path.read_text())
seed_meta = manifest["seeds"][args.seed_index]
raw_dir = Path(seed_meta.get("raw_h5_dir", "raw"))
if not raw_dir.is_absolute():
    raw_dir = WORK / "artifacts/seeds" / raw_dir
H5 = raw_dir / seed_meta["raw_h5"]
TRAJ = seed_meta["traj_key"]
HOUSE_ID = int(seed_meta["house_id"])

import h5py
import numpy as np

with h5py.File(H5) as h:
    g = h[TRAJ]
    obs_scene = json.loads(bytes(g["obs_scene"][()]).rstrip(b"\0"))
    recorded_commanded_actions = [
        json.loads(bytes(x).rstrip(b"\0") or b"{}") for x in g["actions/commanded_action"][:]
    ]
    assert recorded_commanded_actions[0] == {}, (
        "reset observation should carry empty previous action"
    )
    actions = recorded_commanded_actions[1:]
    phases = np.asarray(g["obs/extra/policy_phase"][:], dtype=int)
    seed_obj = np.asarray(g["obs/extra/obj_start"][0], dtype=float)
    seed_base = np.asarray(g["obs/extra/robot_base_pose"][0], dtype=float)
    seed_panda_all = np.asarray(g["env_states/articulations/panda"][:], dtype=float)
    seed_panda = seed_panda_all[0]
    orig_success = bool(g["success"][-1])
log(
    "selected HOM PNP seed={:02d} house={} {}: T={} object={} place={} orig_success={}".format(
        args.seed_index,
        HOUSE_ID,
        TRAJ,
        len(actions),
        obs_scene.get("object_name"),
        obs_scene.get("place_receptacle_name"),
        orig_success,
    )
)

if "MOLMOSPACES_NLTK_DATA" in os.environ:
    os.environ.setdefault("NLTK_DATA", os.environ["MOLMOSPACES_NLTK_DATA"])
import nltk

_nltk_download = nltk.download
nltk.download = lambda *a, **k: True
try:
    import molmo_spaces.utils.synset_utils
finally:
    nltk.download = _nltk_download

from scripts.benchmarks.create_json_benchmark import (
    extract_frozen_config,
    frozen_config_to_episode_spec,
)

frozen = extract_frozen_config(obs_scene)
spec = frozen_config_to_episode_spec(
    frozen_config=frozen,
    obs_scene=obs_scene,
    house_id=HOUSE_ID,
    scene_dataset="procthor-objaverse",
    data_split="val",
    source_h5_file=str(H5),
    source_traj_key=TRAJ,
    source_episode_length=len(recorded_commanded_actions),
    img_resolution=(624, 352),
    camera_system_class=type(frozen.camera_config).__name__
    if hasattr(frozen, "camera_config")
    else None,
    task_horizon_sec=30,
)
spec.task["task_cls"] = spec.task["task_cls"].replace("mujoco_thor.", "molmo_spaces.", 1)
spec.task["task_type"] = "pick_and_place"
spec.task["max_place_receptacle_pos_displacement"] = 0.15
spec.task["max_place_receptacle_rot_displacement"] = float(np.radians(60))

from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.configs.policy_configs import DummyPolicyConfig
from molmo_spaces.policy.dummy_policy import DummyPolicy


class EvalCfg(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: DummyPolicyConfig = DummyPolicyConfig(policy_cls=DummyPolicy)
    task_horizon: int = 500


cfg = EvalCfg()
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

log("constructing JsonEvalTaskSampler")
sampler = JsonEvalTaskSampler(exp_config=cfg, episode_spec=spec)
task = sampler.sample_task(house_index=HOUSE_ID)
obs, info = task.reset()
log(f"task built/reset: {type(task).__name__}")

env = task.env
model = env.current_model
data = env.mj_datas[0] if hasattr(env, "mj_datas") else env.current_data
pickup = spec.task.get("pickup_obj_name") or obs_scene.get("object_name")
place = spec.task.get("place_receptacle_name") or obs_scene.get("place_receptacle_name")
bid = model.body(pickup).id
rid = model.body(place).id
robot = env.current_robot
from molmo_spaces.utils.pose import pose_mat_to_7d

actual_obj = np.r_[data.xpos[bid], data.xquat[bid]]
actual_base = np.asarray(pose_mat_to_7d(robot.robot_view.base.pose), dtype=float)
arm = np.asarray(robot.robot_view.get_move_group("arm").joint_pos, dtype=float)
grip = np.asarray(robot.robot_view.get_move_group("gripper").joint_pos, dtype=float)
obj_err = float(np.max(np.abs(actual_obj - seed_obj)))
base_err = float(np.max(np.abs(actual_base - seed_base)))
arm_err = float(np.max(np.abs(arm - seed_panda[:7])))
grip_err = float(np.max(np.abs(grip - seed_panda[7:9])))
log(
    f"INITIAL_GATE obj_maxerr={obj_err:.3e} base_maxerr={base_err:.3e} arm_maxerr={arm_err:.3e} gripper_maxerr={grip_err:.3e}"
)
if obj_err > 2e-4 or base_err > 2e-4 or arm_err > 2e-4 or grip_err > 2e-4:
    raise RuntimeError("initial-state gate failed")


def body_pose_rel(body_id: int) -> np.ndarray:
    world = np.eye(4, dtype=float)
    world[:3, :3] = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    world[:3, 3] = np.asarray(data.xpos[body_id], dtype=float)
    return np.linalg.inv(np.asarray(robot.robot_view.base.pose, dtype=float)) @ world


def current_arm_joint_dict() -> dict:
    return {"arm": np.asarray(robot.robot_view.get_move_group("arm").joint_pos, dtype=float)}


pickup_pose_rel_observations = [body_pose_rel(bid)]
place_pose_rel_observations = [body_pose_rel(rid)]
target_pose_rel_actions = []
trace = []
first_success = -1
prev_success = bool(task.judge_success())
prev_phase = int(phases[0]) if len(phases) else -1
log(f"collecting datagen info while replaying {len(actions)} actions")
for action_i, act in enumerate(actions):
    obs_i = action_i + 1
    obs, reward, terminal, truncated, infos = task.step(act)
    success = bool(task.judge_success())
    phase = int(phases[obs_i]) if obs_i < len(phases) else -1
    if phase != prev_phase:
        log(f"obs={obs_i} phase={prev_phase}->{phase} success={success}")
        prev_phase = phase
    if first_success < 0 and success:
        first_success = obs_i
    pickup_pose_rel_observations.append(body_pose_rel(bid))
    place_pose_rel_observations.append(body_pose_rel(rid))
    commanded_poses = robot.kinematics.fk(current_arm_joint_dict(), np.eye(4), rel_to_base=True)
    target_pose_rel_actions.append(np.asarray(commanded_poses["arm"], dtype=float))
    info0 = task.get_info()[0]
    qpos = np.r_[
        np.asarray(robot.robot_view.get_move_group("arm").joint_pos, dtype=float),
        np.asarray(robot.robot_view.get_move_group("gripper").joint_pos, dtype=float),
    ]
    row = {
        "action_i": action_i,
        "obs_i": obs_i,
        "phase": phase,
        "success": success,
        "position_error": float(info0.get("position_error", -1)),
        "supported_by_receptacle": bool(info0.get("supported_by_receptacle", False)),
        "robot_contact": bool(info0.get("robot_contact", False))
        if "robot_contact" in info0
        else None,
        "qerr_same": float(np.max(np.abs(qpos - seed_panda_all[obs_i, :9]))),
    }
    trace.append(row)
    if success != prev_success:
        log(
            "SUCCESS_TRANSITION obs={} {}->{} poserr={:.6f} supported={} robot_contact={}".format(
                obs_i,
                prev_success,
                success,
                row["position_error"],
                row["supported_by_receptacle"],
                row["robot_contact"],
            )
        )
        prev_success = success
final = bool(task.judge_success())
success_persistent = first_success >= 0 and all(
    r["success"] for r in trace if r["obs_i"] >= first_success
)
log(
    f"VERDICT first_success={first_success} final={final} success_persistent={success_persistent} original={orig_success}"
)

out = Path(args.out_root) / f"seed_{args.seed_index:02d}"
out.mkdir(parents=True, exist_ok=True)
np.save(
    out / "pickup_obj_pose_rel_observations.npy",
    np.asarray(pickup_pose_rel_observations, dtype=np.float32),
)
np.save(
    out / "place_receptacle_pose_rel_observations.npy",
    np.asarray(place_pose_rel_observations, dtype=np.float32),
)
np.save(out / "target_pose_rel_actions.npy", np.asarray(target_pose_rel_actions, dtype=np.float32))
summary = {
    "seed": f"seed_{args.seed_index:02d}",
    "house_id": HOUSE_ID,
    "traj": TRAJ,
    "task_description": obs_scene.get("task_description"),
    "pickup_obj_name": pickup,
    "place_receptacle_name": place,
    "actions_replayed": len(actions),
    "first_success_obs": first_success,
    "final_success": final,
    "original_success": orig_success,
    "success_persistent_to_end": success_persistent,
    "initial_gate": {
        "obj_maxerr": obj_err,
        "base_maxerr": base_err,
        "arm_maxerr": arm_err,
        "gripper_maxerr": grip_err,
    },
    "trajectory_fidelity": {
        "max_joint_error_rad": max(r["qerr_same"] for r in trace),
        "final_joint_error_rad": trace[-1]["qerr_same"],
    },
    "saved_arrays": {
        "pickup_obj_pose_rel_observations": list(np.asarray(pickup_pose_rel_observations).shape),
        "place_receptacle_pose_rel_observations": list(
            np.asarray(place_pose_rel_observations).shape
        ),
        "target_pose_rel_actions": list(np.asarray(target_pose_rel_actions).shape),
    },
}
(out / "datagen_info_collection_result.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False)
)
(out / "replay_result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
(out / "step_trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False))
log(f"saved datagen arrays and replay result: {out}")
try:
    task.close()
except Exception as e:
    log(f"close warning: {type(e).__name__}: {e}")
raise SystemExit(0 if final and success_persistent else 2)
