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
WORK = Path(
    os.environ.get("MOLMOSPACES_PNP_WORKDIR", str(ROOT / "runtime/mimicgen_pick_and_place"))
)
T0 = time.monotonic()


def log(s):
    print(f"[{time.monotonic() - T0:7.1f}s] {s}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--seed-index", type=int, default=0)
ap.add_argument("--out-name", default="gen_000_same_init")
ap.add_argument("--save-videos", action="store_true")
ap.add_argument("--interp", type=int, default=0, help="MimicGen interpolation steps per subtask")
ap.add_argument("--fixed", type=int, default=0, help="MimicGen fixed steps per subtask")
ap.add_argument("--noise", type=float, default=0.0)
ap.add_argument(
    "--rollout-action-type",
    choices=("joint_position", "tcp_delta"),
    default="joint_position",
    help="control representation used to execute generated EEF waypoints",
)
ap.add_argument(
    "--max-joint-step",
    type=float,
    default=0.12,
    help="maximum absolute arm-joint command change per TCP-delta step (rad)",
)
ap.add_argument(
    "--joint-position-max-step",
    type=float,
    default=0.0,
    help="maximum per-joint command delta for absolute-joint execution; 0 preserves legacy behavior",
)
ap.add_argument(
    "--ik-max-candidate-joint-delta",
    type=float,
    default=0.35,
    help="reject an absolute-IK candidate when any arm joint differs this much from measured q; 0 disables",
)
ap.add_argument("--ik-position-tolerance", type=float, default=0.005, help="bounded pose residual accepted by branch-aware absolute IK")
ap.add_argument("--ik-damping", type=float, default=1e-5, help="DLS damping used by branch-aware absolute IK")
ap.add_argument("--max-tcp-linear-step", type=float, default=0.04)
ap.add_argument("--max-tcp-angular-step", type=float, default=0.15)
ap.add_argument(
    "--tcp-ik-damping",
    type=float,
    default=0.05,
    help="damped-least-squares regularization for TCP-delta differential IK",
)
ap.add_argument(
    "--tcp-joint-limit-avoidance",
    type=float,
    default=0.02,
    help="null-space gain that biases TCP-delta IK away from arm-joint limits",
)
ap.add_argument(
    "--tcp-singularity-threshold",
    type=float,
    default=0.03,
    help="minimum Jacobian singular value below which TCP damping is increased",
)
ap.add_argument(
    "--tcp-stall-max-steps",
    type=int,
    default=24,
    help="fail a TCP waypoint when measured EEF translation remains stationary this many steps",
)
ap.add_argument("--tcp-waypoint-pos-tolerance", type=float, default=0.01)
ap.add_argument("--tcp-waypoint-rot-tolerance", type=float, default=0.08)
ap.add_argument("--tcp-waypoint-max-steps", type=int, default=12)
ap.add_argument(
    "--stop-on-success",
    action="store_true",
    help="terminate generated rollout once task success is reached",
)
ap.add_argument(
    "--source-hdf5",
    default=str(WORK / "artifacts/seeds/robomimic_pnp_10demo_aligned.hdf5"),
    help="MimicGen source HDF5",
)
ap.add_argument(
    "--target-manifest",
    default=str(WORK / "artifacts/seeds/pnp_seed_manifest.json"),
    help="manifest used to build the target MolmoSpaces initial environment",
)
ap.add_argument(
    "--demo-keys",
    default=",".join([f"demo_{i}" for i in range(10)]),
    help="comma-separated source demo keys",
)
ap.add_argument(
    "--select-src-per-subtask",
    action="store_true",
    help="allow MimicGen to choose a different source demo per subtask",
)
ap.add_argument(
    "--mimicgen-rng-seed",
    type=int,
    default=None,
    help="seed NumPy before MimicGen source-subtask selection",
)
ap.add_argument(
    "--omit-final-residual",
    action="store_true",
    help="end task spec at place_success instead of executing post-place residual retreat",
)
ap.add_argument(
    "--post-hold-steps",
    type=int,
    default=0,
    help="after generated waypoints, hold current joint pose for this many steps to verify placement stability",
)
ap.add_argument(
    "--interpolate-from-current-pose",
    action="store_true",
    help="start each subtask interpolation from current robot pose instead of previous target pose",
)
ap.add_argument(
    "--transform-first-robot-pose",
    action="store_true",
    help="include first robot pose for every subtask, not only the first one",
)
args = ap.parse_args()

if args.mimicgen_rng_seed is not None:
    np.random.seed(args.mimicgen_rng_seed)

manifest = json.loads(Path(args.target_manifest).read_text())
seed_meta = manifest["seeds"][args.seed_index]
HOUSE_ID = int(seed_meta["house_id"])
SRC = Path(args.source_hdf5)
DEMO_KEYS = [x.strip() for x in args.demo_keys.split(",") if x.strip()]

import h5py

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

from scripts.benchmarks.create_json_benchmark import (
    extract_frozen_config,
    frozen_config_to_episode_spec,
)

if "episode_spec" in seed_meta:
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec

    spec = EpisodeSpec.model_validate(seed_meta["episode_spec"])
    obs_scene = {
        "object_name": spec.task.get("pickup_obj_name"),
        "place_receptacle_name": spec.task.get("place_receptacle_name"),
        "task_description": spec.language.task_description,
    }
    seed_obj = np.asarray(spec.task["pickup_obj_start_pose"], dtype=float)
    seed_base = np.asarray(spec.task["robot_base_pose"], dtype=float)
    seed_panda = np.asarray(
        list(spec.robot.init_qpos["arm"]) + list(spec.robot.init_qpos["gripper"]), dtype=float
    )
    log(
        f"selected reset-only target={args.seed_index:04d} house={HOUSE_ID} "
        f"layout={seed_meta.get('layout_sha256', 'unknown')[:12]}"
    )
else:
    H5 = WORK / "artifacts/seeds" / seed_meta.get("raw_h5_dir", "raw") / seed_meta["raw_h5"]
    TRAJ = seed_meta["traj_key"]
    with h5py.File(H5) as h:
        g = h[TRAJ]
        obs_scene = json.loads(bytes(g["obs_scene"][()]).rstrip(b"\0"))
        recorded_commanded_actions = [
            json.loads(bytes(x).rstrip(b"\0") or b"{}")
            for x in g["actions/commanded_action"][:]
        ]
        seed_obj = np.asarray(g["obs/extra/obj_start"][0], dtype=float)
        seed_base = np.asarray(g["obs/extra/robot_base_pose"][0], dtype=float)
        seed_panda = np.asarray(g["env_states/articulations/panda"][0], dtype=float)
        orig_success = bool(g["success"][-1])
    task_desc = obs_scene.get("task_description")
    log(
        f"selected PNP seed={args.seed_index:02d} house={HOUSE_ID} {TRAJ} "
        f"task={task_desc!r} orig_success={orig_success}"
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
    M = np.asarray(M, dtype=float)[:3, :3]
    tr = float(np.trace(M))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (M[2, 1] - M[1, 2]) / s
        qy = (M[0, 2] - M[2, 0]) / s
        qz = (M[1, 0] - M[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(M)))
        if i == 0:
            s = np.sqrt(max(0.0, 1.0 + M[0, 0] - M[1, 1] - M[2, 2])) * 2.0
            qw = (M[2, 1] - M[1, 2]) / s if s > 1e-12 else 1.0
            qx = 0.25 * s
            qy = (M[0, 1] + M[1, 0]) / s if s > 1e-12 else 0.0
            qz = (M[0, 2] + M[2, 0]) / s if s > 1e-12 else 0.0
        elif i == 1:
            s = np.sqrt(max(0.0, 1.0 + M[1, 1] - M[0, 0] - M[2, 2])) * 2.0
            qw = (M[0, 2] - M[2, 0]) / s if s > 1e-12 else 1.0
            qx = (M[0, 1] + M[1, 0]) / s if s > 1e-12 else 0.0
            qy = 0.25 * s
            qz = (M[1, 2] + M[2, 1]) / s if s > 1e-12 else 0.0
        else:
            s = np.sqrt(max(0.0, 1.0 + M[2, 2] - M[0, 0] - M[1, 1])) * 2.0
            qw = (M[1, 0] - M[0, 1]) / s if s > 1e-12 else 1.0
            qx = (M[0, 2] + M[2, 0]) / s if s > 1e-12 else 0.0
            qy = (M[1, 2] + M[2, 1]) / s if s > 1e-12 else 0.0
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _quat2mat(q):
    q = np.asarray(q, dtype=float).reshape(4)
    x, y, z, w = q
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


_rs = _types.ModuleType("robosuite")
_rs_utils = _types.ModuleType("robosuite.utils")
_rs_T = _types.ModuleType("robosuite.utils.transform_utils")
_rs_T.mat2quat = _mat2quat
_rs_T.quat2mat = _quat2mat
_rs_utils.transform_utils = _rs_T
_rs.utils = _rs_utils
sys.modules.setdefault("robosuite", _rs)
sys.modules.setdefault("robosuite.utils", _rs_utils)
sys.modules.setdefault("robosuite.utils.transform_utils", _rs_T)
# MimicGen file_utils imports gdown for optional dataset downloads; this run only parses a local HDF5.
# Provide a small import shim so MolmoSpaces .venv can execute local datagen without installing packages.
_gdown = _types.ModuleType("gdown")


def _unused_gdown_download(*args, **kwargs):
    raise RuntimeError("gdown download is not available in this local-only datagen smoke")


_gdown.download = _unused_gdown_download
sys.modules.setdefault("gdown", _gdown)
from mimicgen.env_interfaces.base import MG_EnvInterface
from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator

# MolmoSpaces adapter emits absolute joint targets, not normalized delta actions.
# MimicGen WaypointTrajectory.execute clips arm actions to [-1, 1] whenever noise is not None,
# which is correct for many robosuite controllers but wrong for MolmoSpaces joint targets.
# Patch only this script runtime: skip clipping when waypoint.noise == 0, still add/clip if nonzero noise is requested.
from mimicgen.datagen.waypoint import WaypointTrajectory as _MGWaypointTrajectory


def _molmospaces_joint_execute(
    self, env, env_interface, render=False, video_writer=None, video_skip=5, camera_names=None
):
    write_video = video_writer is not None
    video_count = 0
    states, actions, observations, datagen_infos = [], [], [], []
    success = {k: False for k in env.is_success()}
    stalled_tcp_steps = 0
    previous_tcp_position = None
    for subtask_index, seq in enumerate(self.waypoint_sequences):
        for j in range(len(seq)):
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if write_video:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(
                            env.render(
                                mode="rgb_array", height=512, width=512, camera_name=cam_name
                            )
                        )
                    video_writer.append_data(np.concatenate(video_img, axis=1))
                video_count += 1
            waypoint = seq[j]
            max_steps = args.tcp_waypoint_max_steps if args.rollout_action_type == "tcp_delta" else 1
            for waypoint_step in range(max_steps):
                state = env.get_state()["states"]
                obs = env.get_observation()
                action_pose = env_interface.target_pose_to_action(target_pose=waypoint.pose)
                if args.rollout_action_type == "tcp_delta":
                    linear_norm = float(np.linalg.norm(action_pose[:3]))
                    angular_norm = float(np.linalg.norm(action_pose[3:6]))
                    if (
                        waypoint_step > 0
                        and linear_norm <= args.tcp_waypoint_pos_tolerance
                        and angular_norm <= args.tcp_waypoint_rot_tolerance
                    ):
                        break
                    if linear_norm > args.max_tcp_linear_step:
                        action_pose[:3] *= args.max_tcp_linear_step / linear_norm
                    if angular_norm > args.max_tcp_angular_step:
                        action_pose[3:6] *= args.max_tcp_angular_step / angular_norm
                if (
                    waypoint.noise is not None
                    and float(np.asarray(waypoint.noise).reshape(-1)[0]) != 0.0
                ):
                    action_pose = action_pose + waypoint.noise * np.random.randn(*action_pose.shape)
                    action_pose = np.clip(action_pose, -1.0, 1.0)
                play_action = np.concatenate([action_pose, waypoint.gripper_action], axis=0)
                datagen_info = env_interface.get_datagen_info(action=play_action)
                env.step(play_action)
                current_tcp_position = np.asarray(env.interface_current_eef_pose()[:3, 3], dtype=float)
                if previous_tcp_position is not None and np.linalg.norm(
                    current_tcp_position - previous_tcp_position
                ) < 1e-6:
                    stalled_tcp_steps += 1
                else:
                    stalled_tcp_steps = 0
                previous_tcp_position = current_tcp_position
                if args.rollout_action_type == "tcp_delta" and stalled_tcp_steps >= args.tcp_stall_max_steps:
                    raise RuntimeError(
                        f"TCP rollout stalled for {stalled_tcp_steps} steps; refusing to extend a frozen rollout"
                    )
                states.append(state)
                actions.append(play_action)
                observations.append(obs)
                datagen_infos.append(datagen_info)
                env.executed_waypoint_poses.append(np.asarray(waypoint.pose, dtype=np.float32))
                env.executed_waypoint_subtasks.append(subtask_index)
                cur_success_metrics = env.is_success()
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]
                if args.stop_on_success and bool(cur_success_metrics.get("task", False)):
                    return dict(
                        states=states,
                        observations=observations,
                        datagen_infos=datagen_infos,
                        actions=np.array(actions),
                        success=bool(success["task"]),
                    )
    return dict(
        states=states,
        observations=observations,
        datagen_infos=datagen_infos,
        actions=np.array(actions),
        success=bool(success["task"]),
    )


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
        self.place = spec.task.get("place_receptacle_name") or obs_scene.get(
            "place_receptacle_name"
        )
        self.bid = None
        self.rid = None
        self.step_count = 0
        self.first_success = -1
        self.success_trace = []
        self.executed_joint_commands = []
        self.actual_joint_states = []
        self.actual_eef_states = []
        self.executed_waypoint_poses = []
        self.executed_waypoint_subtasks = []

    def _bind(self):
        self.env = self.task.env
        self.model = self.env.current_model
        self.data = self.env.mj_datas[0] if hasattr(self.env, "mj_datas") else self.env.current_data
        self.robot = self.env.current_robot
        self.bid = self.model.body(self.pickup).id
        self.rid = self.model.body(self.place).id

    def reset(self):
        if self.task is not None:
            try:
                self.task.close()
            except Exception:
                pass
        self.task = self.sampler.sample_task(house_index=HOUSE_ID)
        obs, info = self.task.reset()
        self._bind()
        self.step_count = 0
        self.first_success = -1
        self.success_trace = []
        self.executed_joint_commands = []
        self.actual_joint_states = []
        self.actual_eef_states = []
        self.executed_waypoint_poses = []
        self.executed_waypoint_subtasks = []
        self.ik_candidate_max_joint_deltas = []
        self.ik_continuity_rejections = []
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
        robot_view = self.robot.robot_view
        arm_group = robot_view.get_move_group("arm")
        current_arm = np.asarray(arm_group.joint_pos, dtype=float)
        if action.shape[0] == 7 and args.rollout_action_type == "tcp_delta":
            q0 = robot_view.get_qpos_dict()
            gripper_mgs = set(robot_view.get_gripper_movegroup_ids())
            unlocked = [x for x in robot_view.move_group_ids() if x not in gripper_mgs]
            current_eef = self.interface_current_eef_pose()
            # target_pose_to_action returns a body-frame twist. Convert it once to
            # world coordinates and keep twist_frame aligned with that representation.
            twist_world = np.r_[
                current_eef[:3, :3] @ action[:3],
                current_eef[:3, :3] @ action[3:6],
            ]
            # Solve locally so rank loss is a bounded, logged degradation rather
            # than an unregularized controller branch. The primary Cartesian task
            # remains DLS; only the damping grows close to a singularity.
            jacobian = np.asarray(robot_view.get_jacobian("arm", unlocked), dtype=float)
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            min_singular = float(singular_values[-1]) if singular_values.size else 0.0
            effective_damping = max(
                float(args.tcp_ik_damping),
                max(0.0, float(args.tcp_singularity_threshold) - min_singular),
            )
            qdot = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + effective_damping * np.eye(jacobian.shape[0]),
                twist_world,
            )
            if not np.all(np.isfinite(qdot)):
                raise RuntimeError("non-finite differential-IK joint delta")
            joint_limits = np.asarray(arm_group.joint_pos_limits, dtype=float)
            joint_mid = 0.5 * (joint_limits[:, 0] + joint_limits[:, 1])
            joint_half_range = 0.5 * (joint_limits[:, 1] - joint_limits[:, 0])
            limit_gradient = -(current_arm - joint_mid) / np.maximum(joint_half_range, 1e-6)
            damped_inverse = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + effective_damping * np.eye(jacobian.shape[0]),
                np.eye(jacobian.shape[0]),
            )
            qdot += args.tcp_joint_limit_avoidance * (
                np.eye(jacobian.shape[1]) - damped_inverse @ jacobian
            ) @ limit_gradient
            max_abs = float(np.max(np.abs(qdot))) if qdot.size else 0.0
            if max_abs > args.max_joint_step:
                qdot *= args.max_joint_step / max_abs
            command_arm = arm_group.integrate_joint_vel(current_arm, qdot)
            command_arm = np.clip(
                command_arm,
                joint_limits[:, 0],
                joint_limits[:, 1],
            )
            grip_command = float(action[6])
        elif action.shape[0] == 8 and args.rollout_action_type == "joint_position":
            command_arm = action[:7]
            if args.joint_position_max_step > 0:
                command_arm = current_arm + np.clip(
                    command_arm - current_arm,
                    -args.joint_position_max_step,
                    args.joint_position_max_step,
                )
            grip_command = float(action[7])
        else:
            raise RuntimeError(
                f"rollout_action_type={args.rollout_action_type} received action shape {action.shape}"
            )
        act = {"arm": command_arm.astype(float).tolist(), "gripper": [grip_command]}
        self.executed_joint_commands.append(np.r_[command_arm, grip_command])
        obs, reward, terminal, truncated, infos = self.task.step(act)
        actual_arm = np.asarray(arm_group.joint_pos, dtype=float)
        actual_grip = np.asarray(robot_view.get_move_group("gripper").joint_pos, dtype=float)
        self.actual_joint_states.append(np.r_[actual_arm, actual_grip])
        self.actual_eef_states.append(self.interface_current_eef_pose())
        self.step_count += 1
        succ = bool(self.task.judge_success())
        if succ and self.first_success < 0:
            self.first_success = self.step_count
        info0 = self.task.get_info()[0]
        self.success_trace.append(
            {
                "step": self.step_count,
                "success": succ,
                "position_error": float(info0.get("position_error", -1)),
                "supported_by_receptacle": bool(info0.get("supported_by_receptacle", False)),
                "robot_contact": bool(info0.get("robot_contact", False))
                if "robot_contact" in info0
                else None,
            }
        )
        return (
            self.get_observation(),
            float(np.asarray(reward).reshape(-1)[0]),
            bool(terminal or truncated),
            infos,
        )

    def reset_to(self, state):
        # Not needed by MimicGen generate path for this adapter.
        return self.reset()

    def render(self, mode="human", height=None, width=None, camera_name=None):
        # We rely on MolmoSpaces observation cache for final evidence videos. This hook is only for MimicGen optional writer.
        raw = (
            self.task.observation_cache[-1][0] if self.task and self.task.observation_cache else {}
        )
        if mode == "rgb_array":
            for v in raw.values() if isinstance(raw, dict) else []:
                if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] in (3, 4):
                    return v[..., :3]
            return np.zeros((height or 352, width or 624, 3), dtype=np.uint8)
        return None

    def get_observation(self):
        arm = np.asarray(self.robot.robot_view.get_move_group("arm").joint_pos, dtype=np.float32)
        grip = np.asarray(
            self.robot.robot_view.get_move_group("gripper").joint_pos, dtype=np.float32
        )
        tcp = self.interface_current_eef_pose()
        return {"joint_pos": arm, "gripper_qpos": grip, "eef_pose": tcp.astype(np.float32)}

    def get_state(self):
        arm = np.asarray(self.robot.robot_view.get_move_group("arm").joint_pos, dtype=np.float32)
        grip = np.asarray(
            self.robot.robot_view.get_move_group("gripper").joint_pos, dtype=np.float32
        )
        return {"states": np.r_[arm, grip].astype(np.float32)}

    def get_reward(self):
        return 1.0 if self.is_success()["task"] else 0.0

    def get_goal(self):
        return {}

    def set_goal(self, **kwargs):
        return None

    def is_done(self):
        return False

    def is_success(self):
        return {"task": bool(self.task.judge_success())}

    @property
    def action_dimension(self):
        return 7 if args.rollout_action_type == "tcp_delta" else 8

    @property
    def name(self):
        return "MolmoSpacesPickAndPlaceEnv"

    @property
    def type(self):
        return EnvType.GYM_TYPE

    def serialize(self):
        return {"env_name": self.name, "env_kwargs": {"house_id": HOUSE_ID}, "type": int(self.type)}

    @classmethod
    def create_for_data_processing(cls, *a, **k):
        return cls()

    def interface_current_eef_pose(self):
        poses = self.robot.kinematics.fk(
            {"arm": np.asarray(self.robot.robot_view.get_move_group("arm").joint_pos, dtype=float)},
            np.eye(4),
            rel_to_base=True,
        )
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
        if args.rollout_action_type == "tcp_delta":
            from molmo_spaces.utils.linalg_utils import transform_to_twist

            current_pose = self.get_robot_eef_pose()
            linear, angular = transform_to_twist(
                np.linalg.inv(current_pose) @ np.asarray(target_pose, dtype=float)
            )
            return np.concatenate([linear, angular]).astype(np.float32)
        robot_view = self.env.robot.robot_view
        kinematics = self.env.robot.kinematics
        gripper_mgs = set(robot_view.get_gripper_movegroup_ids())
        mgs_except_gripper = [x for x in robot_view.move_group_ids() if x not in gripper_mgs]
        # Solve from the latest measured state. The solver mutates robot_view while
        # iterating, so restore it before returning a command; otherwise it can bypass
        # the simulator control path and hide an IK branch switch.
        q_measured = {k: np.asarray(v, dtype=float).copy() for k, v in robot_view.get_qpos_dict().items()}
        try:
            jp = kinematics.ik(
                "arm",
                np.asarray(target_pose, dtype=float),
                mgs_except_gripper,
                q_measured,
                robot_view.base.pose,
                rel_to_base=True,
                eps=float(args.ik_position_tolerance),
                damping=float(args.ik_damping),
            )
        finally:
            robot_view.set_qpos_dict(q_measured)
        if jp is None:
            raise RuntimeError("absolute IK failed from measured joint state")
        candidate = np.asarray(jp["arm"], dtype=float)
        max_delta = float(np.max(np.abs(candidate - q_measured["arm"]))) if candidate.size else 0.0
        self.env.ik_candidate_max_joint_deltas.append(max_delta)
        if args.ik_max_candidate_joint_delta > 0 and max_delta > args.ik_max_candidate_joint_delta:
            self.env.ik_continuity_rejections.append({
                "max_joint_delta": max_delta,
                "threshold": float(args.ik_max_candidate_joint_delta),
            })
            raise RuntimeError(
                "absolute IK continuity rejection: "
                f"candidate max joint delta {max_delta:.6f} exceeds "
                f"{args.ik_max_candidate_joint_delta:.6f} from measured state"
            )
        return candidate.astype(np.float32)

    def action_to_target_pose(self, action, relative=True):
        action = np.asarray(action, dtype=float).reshape(-1)
        if args.rollout_action_type == "tcp_delta":
            from molmo_spaces.utils.linalg_utils import twist_to_transform

            return self.get_robot_eef_pose() @ twist_to_transform(action[:3], action[3:6])
        # Joint-position action: convert arm joints to target TCP pose in robot-base frame.
        poses = self.env.robot.kinematics.fk({"arm": action[:7]}, np.eye(4), rel_to_base=True)
        return np.asarray(poses["arm"], dtype=float)

    def action_to_gripper_action(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        return np.asarray(action[-1:], dtype=np.float32)

    def get_object_poses(self):
        return {
            "pickup_obj": self.env.body_pose_rel(self.env.bid),
            "place_receptacle": self.env.body_pose_rel(self.env.rid),
        }

    def get_subtask_term_signals(self):
        return {
            "pregrasp_done": 0,
            "grasp_done": 0,
            "gripper_closed": 0,
            "lift_done": 0,
            "preplace_done": 0,
            "place_success": int(self.env.is_success()["task"]),
        }


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
    subtasks.extend(
        [
            ("place_receptacle", "place_success"),
            ("place_receptacle", None),
        ]
    )
for object_ref, signal in subtasks:
    task_spec.add_subtask(
        object_ref=object_ref,
        subtask_term_signal=signal,
        subtask_term_offset_range=(0, 0),
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
        grip = np.asarray(
            env.robot.robot_view.get_move_group("gripper").joint_pos, dtype=np.float32
        )
        hold_grip = (
            float(results["actions"][-1, -1])
            if len(results["actions"])
            else float(grip.reshape(-1)[0])
        )
        hold_action = (
            np.r_[np.zeros(6, dtype=np.float32), hold_grip]
            if args.rollout_action_type == "tcp_delta"
            else np.r_[arm, hold_grip]
        ).astype(np.float32)
        for _ in range(int(args.post_hold_steps)):
            env.step(hold_action)
            post_hold_actions.append(hold_action.copy())
            env.executed_waypoint_poses.append(env.interface_current_eef_pose().astype(np.float32))
            env.executed_waypoint_subtasks.append(-1)
        if len(post_hold_actions):
            results["actions"] = np.concatenate(
                [
                    np.asarray(results["actions"], dtype=np.float32),
                    np.asarray(post_hold_actions, dtype=np.float32),
                ],
                axis=0,
            )
    final_success = bool(env.task.judge_success())
    persistent = env.first_success >= 0 and all(
        r["success"] for r in env.success_trace if r["step"] >= env.first_success
    )
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
        "mimicgen_rng_seed": args.mimicgen_rng_seed,
        "interpolate_from_last_target_pose": (not bool(args.interpolate_from_current_pose)),
        "transform_first_robot_pose": bool(args.transform_first_robot_pose),
        "stop_on_success": bool(args.stop_on_success),
        "omit_final_residual": bool(args.omit_final_residual),
        "post_hold_steps": int(args.post_hold_steps),
        "rollout_action_type": args.rollout_action_type,
        "max_joint_step": float(args.max_joint_step),
        "joint_position_max_step": float(args.joint_position_max_step),
        "ik_max_candidate_joint_delta": float(args.ik_max_candidate_joint_delta),
        "ik_position_tolerance": float(args.ik_position_tolerance),
        "ik_damping": float(args.ik_damping),
        "ik_candidate_max_joint_delta": float(max(env.ik_candidate_max_joint_deltas, default=0.0)),
        "ik_continuity_rejection_count": int(len(env.ik_continuity_rejections)),
        "ik_continuity_rejections": env.ik_continuity_rejections,
        "max_tcp_linear_step": float(args.max_tcp_linear_step),
        "max_tcp_angular_step": float(args.max_tcp_angular_step),
        "tcp_ik_damping": float(args.tcp_ik_damping),
        "tcp_joint_limit_avoidance": float(args.tcp_joint_limit_avoidance),
        "tcp_singularity_threshold": float(args.tcp_singularity_threshold),
        "tcp_stall_max_steps": int(args.tcp_stall_max_steps),
        "tcp_waypoint_pos_tolerance": float(args.tcp_waypoint_pos_tolerance),
        "tcp_waypoint_rot_tolerance": float(args.tcp_waypoint_rot_tolerance),
        "tcp_waypoint_max_steps": int(args.tcp_waypoint_max_steps),
        "generated_success_any": bool(results["success"]),
        "final_success": final_success,
        "first_success_step": int(env.first_success),
        "success_persistent_to_end": bool(persistent),
        "num_actions_executed": int(len(results["actions"])),
        "src_demo_inds": [int(x) for x in results["src_demo_inds"]],
        "distinct_src_demo_count": len({int(x) for x in results["src_demo_inds"]}),
        "task_spec": json.loads(task_spec.serialize()),
        "notes": "Full-rollout PnP generation from the configured source dataset; stop_on_success must be false for acceptance.",
    }
    (out / "generate_result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "success_trace.json").write_text(
        json.dumps(env.success_trace, indent=2, ensure_ascii=False)
    )
    generated_actions = np.asarray(results["actions"], dtype=np.float32)
    waypoint_poses = np.asarray(env.executed_waypoint_poses, dtype=np.float32)
    waypoint_subtasks = np.asarray(env.executed_waypoint_subtasks, dtype=np.int32)
    joint_commands = np.asarray(env.executed_joint_commands, dtype=np.float32)
    actual_joints = np.asarray(env.actual_joint_states, dtype=np.float32)
    actual_eef = np.asarray(env.actual_eef_states, dtype=np.float32)
    src_labels = np.asarray(results["src_demo_labels"], dtype=np.int32).reshape(-1)
    if len(post_hold_actions):
        src_labels = np.concatenate(
            [src_labels, -np.ones(len(post_hold_actions), dtype=np.int32)]
        )
    if not (
        len(generated_actions)
        == len(waypoint_poses)
        == len(waypoint_subtasks)
        == len(joint_commands)
        == len(actual_joints)
        == len(actual_eef)
        == len(src_labels)
    ):
        raise RuntimeError(
            "generated persistence alignment mismatch: "
            f"actions={len(generated_actions)} waypoints={len(waypoint_poses)} "
            f"subtasks={len(waypoint_subtasks)} commands={len(joint_commands)} "
            f"actual_joints={len(actual_joints)} actual_eef={len(actual_eef)} "
            f"source_labels={len(src_labels)}"
        )
    np.save(out / "generated_actions.npy", generated_actions)
    np.save(out / "executed_waypoint_poses.npy", waypoint_poses)
    np.save(out / "executed_waypoint_subtasks.npy", waypoint_subtasks)
    np.save(out / "executed_joint_commands.npy", joint_commands)
    np.save(out / "actual_joint_states.npy", actual_joints)
    np.save(out / "actual_eef_states.npy", actual_eef)
    np.save(out / "source_demo_labels.npy", src_labels)
    with h5py.File(out / "generated_replay_package.hdf5", "w") as replay_h5:
        data = replay_h5.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=generated_actions)
        demo.create_dataset("executed_waypoint_poses", data=waypoint_poses)
        demo.create_dataset("executed_waypoint_subtasks", data=waypoint_subtasks)
        demo.create_dataset("executed_joint_commands", data=joint_commands)
        demo.create_dataset("actual_joint_states", data=actual_joints)
        demo.create_dataset("actual_eef_states", data=actual_eef)
        demo.create_dataset("source_demo_labels", data=src_labels)
        demo.attrs["num_samples"] = len(generated_actions)
        demo.attrs["rollout_action_type"] = args.rollout_action_type
        demo.attrs["target_manifest"] = str(Path(args.target_manifest))
        demo.attrs["target_seed_index"] = int(args.seed_index)
        demo.attrs["source_hdf5"] = str(SRC)
        data.attrs["total"] = len(generated_actions)
    if args.save_videos:
        from molmo_spaces.utils.save_utils import save_videos_from_raw_observations

        raw = [batch[0] for batch in env.task.observation_cache]
        save_videos_from_raw_observations(
            raw,
            save_dir=str(out),
            fps=1000 / 66,
            episode_idx=0,
            save_file_suffix="_mimicgen_pnp_gen000",
            sensor_suite=env.task.sensor_suite,
        )
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if final_success and persistent else 2)
finally:
    try:
        env.close()
    except Exception as e:
        log(f"close warning: {type(e).__name__}: {e}")
