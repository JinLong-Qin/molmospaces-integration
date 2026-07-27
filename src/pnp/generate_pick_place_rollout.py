from __future__ import annotations
import argparse, json, os, signal, sys, time
from pathlib import Path
import numpy as np

signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("hard timeout")))
signal.alarm(2400)
ROOT = Path(os.environ.get("MOLMOSPACES_ROOT", "."))
MIMICGEN_ROOT = Path(os.environ.get("MIMICGEN_ROOT", "vendor/mimicgen"))
ROBOMIMIC_ROOT = Path(os.environ.get("ROBOMIMIC_ROOT", "vendor/robomimic"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MIMICGEN_ROOT))
sys.path.insert(0, str(ROBOMIMIC_ROOT))
WORK = Path(os.environ.get("MOLMOSPACES_PNP_WORKDIR", str(ROOT / "runtime/mimicgen_pick_and_place")))
T0 = time.monotonic()
def log(s): print(f"[{time.monotonic()-T0:7.1f}s] {s}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--seed-index", type=int, default=0)
ap.add_argument("--out-name", default="gen_000_same_init")
ap.add_argument("--save-videos", action="store_true")
ap.add_argument("--interp", type=int, default=0, help="MimicGen interpolation steps per subtask")
ap.add_argument("--fixed", type=int, default=0, help="MimicGen fixed steps per subtask")
ap.add_argument("--noise", type=float, default=0.0)
ap.add_argument("--stop-on-success", action="store_true", help="terminate generated rollout once task success is reached")
ap.add_argument("--source-hdf5", default=str(WORK / "artifacts/seeds/robomimic_pnp_10demo_aligned.hdf5"), help="MimicGen source HDF5")
ap.add_argument("--target-manifest", default=str(WORK / "artifacts/seeds/pnp_seed_manifest.json"), help="manifest used to build the target MolmoSpaces initial environment")
ap.add_argument("--demo-keys", default=",".join([f"demo_{i}" for i in range(10)]), help="comma-separated source demo keys")
ap.add_argument("--select-src-per-subtask", action="store_true", help="allow MimicGen to choose a different source demo per subtask")
ap.add_argument("--omit-final-residual", action="store_true", help="end task spec at place_success instead of executing post-place residual retreat")
ap.add_argument("--post-hold-steps", type=int, default=0, help="after generated waypoints, hold current joint pose for this many steps to verify placement stability")
ap.add_argument("--interpolate-from-current-pose", action="store_true", help="start each subtask interpolation from current robot pose instead of previous target pose")
ap.add_argument("--transform-first-robot-pose", action="store_true", help="include first robot pose for every subtask, not only the first one")
args = ap.parse_args()

manifest = json.loads(Path(args.target_manifest).read_text())
seed_meta = manifest["seeds"][args.seed_index]
H5 = WORK / "artifacts/seeds" / seed_meta.get("raw_h5_dir", "raw") / seed_meta["raw_h5"]
TRAJ = seed_meta["traj_key"]
HOUSE_ID = int(seed_meta["house_id"])
SRC = Path(args.source_hdf5)
DEMO_KEYS = [x.strip() for x in args.demo_keys.split(",") if x.strip()]

import h5py
with h5py.File(H5) as h:
    g = h[TRAJ]
    obs_scene = json.loads(bytes(g["obs_scene"][()]).rstrip(b"\0"))
    recorded_commanded_actions = [json.loads(bytes(x).rstrip(b"\0") or b"{}") for x in g["actions/commanded_action"][:]]
    seed_obj = np.asarray(g["obs/extra/obj_start"][0], dtype=float)
    seed_base = np.asarray(g["obs/extra/robot_base_pose"][0], dtype=float)
    seed_panda = np.asarray(g["env_states/articulations/panda"][0], dtype=float)
    orig_success = bool(g["success"][-1])
task_desc = obs_scene.get("task_description")
log(f"selected PNP seed={args.seed_index:02d} house={HOUSE_ID} {TRAJ} task={task_desc!r} orig_success={orig_success}")

if "MOLMOSPACES_NLTK_DATA" in os.environ:
    os.environ.setdefault("NLTK_DATA", os.environ["MOLMOSPACES_NLTK_DATA"])
try:
    import nltk
    _nltk_download = nltk.download
    nltk.download = lambda *a, **k: True
    try:
        import molmo_spaces.utils.synset_utils
    finally:
        nltk.download = _nltk_download
except ModuleNotFoundError as e:
    if e.name != "nltk":
        raise
    # The MimicGen conda env does not ship nltk, but this script only needs to
    # avoid runtime downloads while importing MolmoSpaces. Continue if the
    # downstream task path does not require nltk at execution time.
    log("nltk not installed in this interpreter; continuing without synset_utils pre-import")

from scripts.benchmarks.create_json_benchmark import extract_frozen_config, frozen_config_to_episode_spec
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
    camera_system_class=type(frozen.camera_config).__name__ if hasattr(frozen, "camera_config") else None,
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
    task_horizon: int = 700
cfg = EvalCfg()
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
from molmo_spaces.utils.pose import pose_mat_to_7d

class EnvType:
    GYM_TYPE = 2
class EnvBase(object):
    pass
# Runtime-only shim: MimicGen pose_utils imports robosuite.utils.transform_utils
# just for mat2quat / quat2mat. MolmoSpaces .venv does not include robosuite/numba,
# so provide the tiny transform_utils surface needed by MimicGen without modifying vendor code.
import types as _types
def _mat2quat(M):
    M = np.asarray(M, dtype=float)[:3,:3]
    tr = float(np.trace(M))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (M[2,1] - M[1,2]) / s
        qy = (M[0,2] - M[2,0]) / s
        qz = (M[1,0] - M[0,1]) / s
    else:
        i = int(np.argmax(np.diag(M)))
        if i == 0:
            s = np.sqrt(max(0.0, 1.0 + M[0,0] - M[1,1] - M[2,2])) * 2.0
            qw = (M[2,1] - M[1,2]) / s if s > 1e-12 else 1.0
            qx = 0.25 * s
            qy = (M[0,1] + M[1,0]) / s if s > 1e-12 else 0.0
            qz = (M[0,2] + M[2,0]) / s if s > 1e-12 else 0.0
        elif i == 1:
            s = np.sqrt(max(0.0, 1.0 + M[1,1] - M[0,0] - M[2,2])) * 2.0
            qw = (M[0,2] - M[2,0]) / s if s > 1e-12 else 1.0
            qx = (M[0,1] + M[1,0]) / s if s > 1e-12 else 0.0
            qy = 0.25 * s
            qz = (M[1,2] + M[2,1]) / s if s > 1e-12 else 0.0
        else:
            s = np.sqrt(max(0.0, 1.0 + M[2,2] - M[0,0] - M[1,1])) * 2.0
            qw = (M[1,0] - M[0,1]) / s if s > 1e-12 else 1.0
            qx = (M[0,2] + M[2,0]) / s if s > 1e-12 else 0.0
            qy = (M[1,2] + M[2,1]) / s if s > 1e-12 else 0.0
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.,0.,0.,1.])
def _quat2mat(q):
    q = np.asarray(q, dtype=float).reshape(4)
    x,y,z,w = q
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    x,y,z,w = q / n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=float)
_rs = _types.ModuleType('robosuite')
_rs_utils = _types.ModuleType('robosuite.utils')
_rs_T = _types.ModuleType('robosuite.utils.transform_utils')
_rs_T.mat2quat = _mat2quat
_rs_T.quat2mat = _quat2mat
_rs_utils.transform_utils = _rs_T
_rs.utils = _rs_utils
sys.modules.setdefault('robosuite', _rs)
sys.modules.setdefault('robosuite.utils', _rs_utils)
sys.modules.setdefault('robosuite.utils.transform_utils', _rs_T)
# MimicGen file_utils imports gdown for optional dataset downloads; this run only parses a local HDF5.
# Provide a small import shim so MolmoSpaces .venv can execute local datagen without installing packages.
_gdown = _types.ModuleType('gdown')
def _unused_gdown_download(*args, **kwargs):
    raise RuntimeError('gdown download is not available in this local-only datagen smoke')
_gdown.download = _unused_gdown_download
sys.modules.setdefault('gdown', _gdown)
from mimicgen.env_interfaces.base import MG_EnvInterface
from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator
# MolmoSpaces adapter emits absolute joint targets, not normalized delta actions.
# MimicGen WaypointTrajectory.execute clips arm actions to [-1, 1] whenever noise is not None,
# which is correct for many robosuite controllers but wrong for MolmoSpaces joint targets.
# Patch only this script runtime: skip clipping when waypoint.noise == 0, still add/clip if nonzero noise is requested.
from mimicgen.datagen.waypoint import WaypointTrajectory as _MGWaypointTrajectory
def _molmospaces_joint_execute(self, env, env_interface, render=False, video_writer=None, video_skip=5, camera_names=None):
    write_video = (video_writer is not None)
    video_count = 0
    states, actions, observations, datagen_infos = [], [], [], []
    success = {k: False for k in env.is_success()}
    for seq in self.waypoint_sequences:
        for j in range(len(seq)):
            if render:
                env.render(mode='human', camera_name=camera_names[0])
            if write_video:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode='rgb_array', height=512, width=512, camera_name=cam_name))
                    video_writer.append_data(np.concatenate(video_img, axis=1))
                video_count += 1
            waypoint = seq[j]
            state = env.get_state()['states']
            obs = env.get_observation()
            action_pose = env_interface.target_pose_to_action(target_pose=waypoint.pose)
            if waypoint.noise is not None and float(np.asarray(waypoint.noise).reshape(-1)[0]) != 0.0:
                action_pose = action_pose + waypoint.noise * np.random.randn(*action_pose.shape)
                action_pose = np.clip(action_pose, -1., 1.)
            play_action = np.concatenate([action_pose, waypoint.gripper_action], axis=0)
            datagen_info = env_interface.get_datagen_info(action=play_action)
            env.step(play_action)
            states.append(state)
            actions.append(play_action)
            observations.append(obs)
            datagen_infos.append(datagen_info)
            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]
            if args.stop_on_success and bool(cur_success_metrics.get('task', False)):
                return dict(states=states, observations=observations, datagen_infos=datagen_infos, actions=np.array(actions), success=bool(success['task']))
    return dict(states=states, observations=observations, datagen_infos=datagen_infos, actions=np.array(actions), success=bool(success['task']))
_MGWaypointTrajectory.execute = _molmospaces_joint_execute

class MolmoSpacesPnpEnv(EnvBase):
    def __init__(self):
        self.sampler = JsonEvalTaskSampler(exp_config=cfg, episode_spec=spec)
        self.task = None
        self.env = None
        self.robot = None
        self.model = None
        self.data = None
        self.pickup = spec.task.get("pickup_obj_name") or obs_scene.get("object_name")
        self.place = spec.task.get("place_receptacle_name") or obs_scene.get("place_receptacle_name")
        self.bid = None
        self.rid = None
        self.step_count = 0
        self.first_success = -1
        self.success_trace = []
    def _bind(self):
        self.env = self.task.env
        self.model = self.env.current_model
        self.data = self.env.mj_datas[0] if hasattr(self.env, "mj_datas") else self.env.current_data
        self.robot = self.env.current_robot
        self.bid = self.model.body(self.pickup).id
        self.rid = self.model.body(self.place).id
    def reset(self):
        if self.task is not None:
            try: self.task.close()
            except Exception: pass
        self.task = self.sampler.sample_task(house_index=HOUSE_ID)
        obs, info = self.task.reset()
        self._bind()
        self.step_count = 0
        self.first_success = -1
        self.success_trace = []
        # strict same-initial-state gate for this first smoke
        actual_obj = np.r_[self.data.xpos[self.bid], self.data.xquat[self.bid]]
        actual_base = np.asarray(pose_mat_to_7d(self.robot.robot_view.base.pose), dtype=float)
        arm = np.asarray(self.robot.robot_view.get_move_group("arm").joint_pos, dtype=float)
        grip = np.asarray(self.robot.robot_view.get_move_group("gripper").joint_pos, dtype=float)
        errs = dict(
            obj=float(np.max(np.abs(actual_obj - seed_obj))),
            base=float(np.max(np.abs(actual_base - seed_base))),
            arm=float(np.max(np.abs(arm - seed_panda[:7]))),
            gripper=float(np.max(np.abs(grip - seed_panda[7:9]))),
        )
        log(f"INITIAL_GATE {errs}")
        if max(errs.values()) > 2e-4:
            raise RuntimeError(f"initial-state gate failed: {errs}")
        return self.get_observation()
    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] != 8:
            raise RuntimeError(f"expected 8D joint+gripper action, got {action.shape}")
        act = {"arm": action[:7].astype(float).tolist(), "gripper": [float(action[7])]}
        obs, reward, terminal, truncated, infos = self.task.step(act)
        self.step_count += 1
        succ = bool(self.task.judge_success())
        if succ and self.first_success < 0:
            self.first_success = self.step_count
        info0 = self.task.get_info()[0]
        self.success_trace.append({
            "step": self.step_count,
            "success": succ,
            "position_error": float(info0.get("position_error", -1)),
            "supported_by_receptacle": bool(info0.get("supported_by_receptacle", False)),
            "robot_contact": bool(info0.get("robot_contact", False)) if "robot_contact" in info0 else None,
        })
        return self.get_observation(), float(np.asarray(reward).reshape(-1)[0]), bool(terminal or truncated), infos
    def reset_to(self, state):
        # Not needed by MimicGen generate path for this adapter.
        return self.reset()
    def render(self, mode="human", height=None, width=None, camera_name=None):
        # We rely on MolmoSpaces observation cache for final evidence videos. This hook is only for MimicGen optional writer.
        raw = self.task.observation_cache[-1][0] if self.task and self.task.observation_cache else {}
        if mode == "rgb_array":
            for v in raw.values() if isinstance(raw, dict) else []:
                if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] in (3,4):
                    return v[..., :3]
            return np.zeros((height or 352, width or 624, 3), dtype=np.uint8)
        return None
    def get_observation(self):
        arm = np.asarray(self.robot.robot_view.get_move_group("arm").joint_pos, dtype=np.float32)
        grip = np.asarray(self.robot.robot_view.get_move_group("gripper").joint_pos, dtype=np.float32)
        tcp = self.interface_current_eef_pose()
        return {"joint_pos": arm, "gripper_qpos": grip, "eef_pose": tcp.astype(np.float32)}
    def get_state(self):
        arm = np.asarray(self.robot.robot_view.get_move_group("arm").joint_pos, dtype=np.float32)
        grip = np.asarray(self.robot.robot_view.get_move_group("gripper").joint_pos, dtype=np.float32)
        return {"states": np.r_[arm, grip].astype(np.float32)}
    def get_reward(self): return 1.0 if self.is_success()["task"] else 0.0
    def get_goal(self): return {}
    def set_goal(self, **kwargs): return None
    def is_done(self): return False
    def is_success(self): return {"task": bool(self.task.judge_success())}
    @property
    def action_dimension(self): return 8
    @property
    def name(self): return "MolmoSpacesPickAndPlaceEnv"
    @property
    def type(self): return EnvType.GYM_TYPE
    def serialize(self): return {"env_name": self.name, "env_kwargs": {"house_id": HOUSE_ID}, "type": int(self.type)}
    @classmethod
    def create_for_data_processing(cls, *a, **k): return cls()
    def interface_current_eef_pose(self):
        poses = self.robot.kinematics.fk({"arm": np.asarray(self.robot.robot_view.get_move_group("arm").joint_pos, dtype=float)}, np.eye(4), rel_to_base=True)
        return np.asarray(poses["arm"], dtype=float)
    def body_pose_rel(self, body_id):
        world = np.eye(4, dtype=float)
        world[:3, :3] = np.asarray(self.data.xmat[body_id], dtype=float).reshape(3, 3)
        world[:3, 3] = np.asarray(self.data.xpos[body_id], dtype=float)
        return np.linalg.inv(np.asarray(self.robot.robot_view.base.pose, dtype=float)) @ world
    def close(self):
        if self.task is not None:
            self.task.close()

class MG_MolmoSpacesPickAndPlace(MG_EnvInterface):
    INTERFACE_TYPE = "molmospaces"
    def get_robot_eef_pose(self):
        return self.env.interface_current_eef_pose()
    def target_pose_to_action(self, target_pose, relative=True):
        robot_view = self.env.robot.robot_view
        kinematics = self.env.robot.kinematics
        gripper_mgs = set(robot_view.get_gripper_movegroup_ids())
        mgs_except_gripper = [x for x in robot_view.move_group_ids() if x not in gripper_mgs]
        jp = kinematics.ik(
            "arm",
            np.asarray(target_pose, dtype=float),
            mgs_except_gripper,
            robot_view.get_qpos_dict(),
            robot_view.base.pose,
            rel_to_base=True,
        )
        if jp is None:
            # Hold current arm pose on IK miss; record via env trace by lack of success / later diagnostics.
            return np.asarray(robot_view.get_move_group("arm").joint_pos, dtype=np.float32)
        return np.asarray(jp["arm"], dtype=np.float32)
    def action_to_target_pose(self, action, relative=True):
        # action is 8D joint+gripper; convert arm joints to target TCP pose in robot-base frame.
        action = np.asarray(action, dtype=float).reshape(-1)
        poses = self.env.robot.kinematics.fk({"arm": action[:7]}, np.eye(4), rel_to_base=True)
        return np.asarray(poses["arm"], dtype=float)
    def action_to_gripper_action(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        return np.asarray(action[7:8], dtype=np.float32)
    def get_object_poses(self):
        return {"pickup_obj": self.env.body_pose_rel(self.env.bid), "place_receptacle": self.env.body_pose_rel(self.env.rid)}
    def get_subtask_term_signals(self):
        return {"pregrasp_done": 0, "grasp_done": 0, "gripper_closed": 0, "lift_done": 0, "preplace_done": 0, "place_success": int(self.env.is_success()["task"])}

# PnP task spec: approach/grasp/lift with pickup object frame, placement/final residual with receptacle frame.
task_spec = MG_TaskSpec()
subtasks = [
    ("pickup_obj", "pregrasp_done"),
    ("pickup_obj", "grasp_done"),
    ("pickup_obj", "gripper_closed"),
    ("pickup_obj", "lift_done"),
    ("place_receptacle", "preplace_done"),
]
if args.omit_final_residual:
    # Use a source dataset already truncated at the stable place-success boundary.
    # MimicGen requires the final subtask signal to be None.
    subtasks.append(("place_receptacle", None))
else:
    subtasks.extend([
        ("place_receptacle", "place_success"),
        ("place_receptacle", None),
    ])
for object_ref, signal in subtasks:
    task_spec.add_subtask(
        object_ref=object_ref,
        subtask_term_signal=signal,
        subtask_term_offset_range=(0,0),
        selection_strategy="random",
        selection_strategy_kwargs=None,
        action_noise=float(args.noise),
        num_interpolation_steps=int(args.interp),
        num_fixed_steps=int(args.fixed),
        apply_noise_during_interpolation=False,
    )

log("constructing DataGenerator")
gen = DataGenerator(task_spec=task_spec, dataset_path=str(SRC), demo_keys=DEMO_KEYS)
env = MolmoSpacesPnpEnv()
iface = MG_MolmoSpacesPickAndPlace(env)
out = WORK / "artifacts/mimicgen_pnp" / args.out_name
out.mkdir(parents=True, exist_ok=True)
try:
    log("running DataGenerator.generate")
    results = gen.generate(
        env=env,
        env_interface=iface,
        select_src_per_subtask=bool(args.select_src_per_subtask),
        transform_first_robot_pose=bool(args.transform_first_robot_pose),
        interpolate_from_last_target_pose=not bool(args.interpolate_from_current_pose),
        render=False,
        video_writer=None,
        video_skip=1,
        camera_names=None,
    )
    post_hold_actions = []
    if int(args.post_hold_steps) > 0:
        arm = np.asarray(env.robot.robot_view.get_move_group("arm").joint_pos, dtype=np.float32)
        grip = np.asarray(env.robot.robot_view.get_move_group("gripper").joint_pos, dtype=np.float32)
        hold_grip = float(results["actions"][-1, 7]) if len(results["actions"]) else float(grip.reshape(-1)[0])
        hold_action = np.r_[arm, hold_grip].astype(np.float32)
        for _ in range(int(args.post_hold_steps)):
            env.step(hold_action)
            post_hold_actions.append(hold_action.copy())
        if len(post_hold_actions):
            results["actions"] = np.concatenate([np.asarray(results["actions"], dtype=np.float32), np.asarray(post_hold_actions, dtype=np.float32)], axis=0)
    final_success = bool(env.task.judge_success())
    persistent = env.first_success >= 0 and all(r["success"] for r in env.success_trace if r["step"] >= env.first_success)
    summary = {
        "out_name": args.out_name,
        "source_hdf5": str(SRC),
        "target_manifest": str(Path(args.target_manifest)),
        "source_demo_keys": DEMO_KEYS,
        "generation_env_seed_index": int(args.seed_index),
        "generation_env_house_id": int(HOUSE_ID),
        "is_mimicgen_generate_call": True,
        "same_initial_state_smoke": bool(args.seed_index == 0),
        "select_src_per_subtask": bool(args.select_src_per_subtask),
        "interpolate_from_last_target_pose": (not bool(args.interpolate_from_current_pose)),
        "transform_first_robot_pose": bool(args.transform_first_robot_pose),
        "stop_on_success": bool(args.stop_on_success),
        "omit_final_residual": bool(args.omit_final_residual),
        "post_hold_steps": int(args.post_hold_steps),
        "generated_success_any": bool(results["success"]),
        "final_success": final_success,
        "first_success_step": int(env.first_success),
        "success_persistent_to_end": bool(persistent),
        "num_actions_executed": int(len(results["actions"])),
        "src_demo_inds": [int(x) for x in results["src_demo_inds"]],
        "task_spec": json.loads(task_spec.serialize()),
        "notes": "Full-rollout PnP generation from the configured source dataset; stop_on_success must be false for acceptance.",
    }
    (out / "generate_result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "success_trace.json").write_text(json.dumps(env.success_trace, indent=2, ensure_ascii=False))
    np.save(out / "generated_actions.npy", np.asarray(results["actions"], dtype=np.float32))
    if args.save_videos:
        from molmo_spaces.utils.save_utils import save_videos_from_raw_observations
        raw = [batch[0] for batch in env.task.observation_cache]
        save_videos_from_raw_observations(raw, save_dir=str(out), fps=1000/66, episode_idx=0, save_file_suffix="_mimicgen_pnp_gen000", sensor_suite=env.task.sensor_suite)
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if final_success and persistent else 2)
finally:
    try: env.close()
    except Exception as e: log(f"close warning: {type(e).__name__}: {e}")
