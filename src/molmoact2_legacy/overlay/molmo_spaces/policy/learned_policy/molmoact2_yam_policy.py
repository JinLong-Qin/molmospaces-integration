"""MolmoAct2-BimanualYAM policy adapter for MolmoSpaces.

This module keeps the MolmoSpaces observation/action layout and only adapts it
to the external MolmoAct2 ``/act`` wire contract:

    top_cam, left_cam, right_cam, instruction, state, optional num_steps

The pure helpers are intentionally small so request/response conversion can be
smoke-tested without loading MolmoAct2 weights or starting a server.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy

log = logging.getLogger(__name__)

MOLMOACT2_YAM_ACTION_DIM = 14
MOLMOACT2_YAM_STATE_ACTION_ORDER = (
    "left_arm_6",
    "left_gripper",
    "right_arm_6",
    "right_gripper",
)
MOLMOACT2_YAM_GRIPPER_MAX = 0.041
MOLMOACT2_YAM_GRIPPER_SCALE_SOURCE = "molmospaces_bimanual_yam_xml_ctrlrange"
MOLMOACT2_YAM_ACTION_GRIPPER_SEMANTICS = "official:1=open,0=closed"
DEFAULT_MOLMOACT2_YAM_INSTRUCTION = "Put everything into the box."

DEFAULT_MOLMOACT2_YAM_CAMERA_MAPPING = {
    "top_cam": "exo_camera",
    "left_cam": "left_wrist_camera",
    "right_cam": "right_wrist_camera",
}


class MolmoAct2HTTPClient:
    """Minimal HTTP client for a MolmoAct2 ``/act`` endpoint.

    The client imports optional HTTP/JSON dependencies lazily so unit and smoke
    tests can inject a fake client without installing server-side packages.
    """

    def __init__(self, endpoint_url: str, request_timeout: float = 60.0, session=None) -> None:
        self.endpoint_url = endpoint_url
        self.request_timeout = request_timeout

        if session is None:
            try:
                import requests
            except ImportError as e:
                raise ImportError(
                    "MolmoAct2HTTPClient requires `requests` for real server calls. "
                    "Inject a fake client for no-server tests."
                ) from e
            session = requests.Session()

        self.session = session

    def act(self, request: dict[str, Any]) -> Any:
        body = _dumps_json_numpy_compatible(request)
        response = self.session.post(
            self.endpoint_url,
            headers={"Content-Type": "application/json"},
            data=body,
            timeout=self.request_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"MolmoAct2 /act error {response.status_code}: {response.text[:500]}"
            )
        return _loads_json_numpy_compatible(response.text, response)

    def close(self) -> None:
        close = getattr(self.session, "close", None)
        if close is not None:
            close()


def _dumps_json_numpy_compatible(payload: dict[str, Any]) -> str:
    try:
        import json_numpy

        return json_numpy.dumps(payload)
    except ImportError:
        return json.dumps(_to_jsonable(payload))


def _loads_json_numpy_compatible(text: str, response: Any) -> Any:
    try:
        import json_numpy

        return json_numpy.loads(text)
    except ImportError:
        return response.json()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    return value


def _array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    summary: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size:
        if np.issubdtype(array.dtype, np.number):
            summary.update(
                {
                    "min": float(np.nanmin(array)),
                    "max": float(np.nanmax(array)),
                    "mean": float(np.nanmean(array)),
                }
            )
        else:
            summary["sample"] = str(array.reshape(-1)[0])
    return summary


def append_molmoact2_yam_raw_action_log(
    path: str | Path,
    *,
    call_index: int,
    request: dict[str, Any],
    response: Any,
    actions: np.ndarray,
    camera_mapping: dict[str, str],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "call_index": int(call_index),
        "instruction": request.get("instruction"),
        "camera_mapping": dict(camera_mapping),
        "state": _to_jsonable(np.asarray(request.get("state"), dtype=np.float32)),
        "state_summary": _array_summary(request.get("state")),
        "request_image_summaries": {
            key: _array_summary(request[key])
            for key in ("top_cam", "left_cam", "right_cam")
            if key in request
        },
        "raw_response_type": type(response).__name__,
        "actions_shape": list(actions.shape),
        "actions": _to_jsonable(actions),
        "first_action": _to_jsonable(actions[0]),
        "last_action": _to_jsonable(actions[-1]),
        "left_gripper_range": [float(np.min(actions[:, 6])), float(np.max(actions[:, 6]))],
        "right_gripper_range": [float(np.min(actions[:, 13])), float(np.max(actions[:, 13]))],
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return output_path


def _write_ppm_image(path: Path, image: np.ndarray) -> None:
    image = _to_uint8_image(image, path.name)
    height, width = image.shape[:2]
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + image.tobytes())


def dump_molmoact2_yam_request_debug_images(
    request: dict[str, Any],
    output_dir: str | Path,
    *,
    call_index: int,
) -> list[Path]:
    """Dump the exact RGB images included in one MolmoAct2 YAM ``/act`` request.

    The writer uses dependency-free binary PPM files so debug output remains
    bounded and does not require image libraries in smoke environments.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"act_{call_index:06d}"
    written: list[Path] = []
    image_metadata: dict[str, Any] = {}
    for request_key in ("top_cam", "left_cam", "right_cam"):
        image = _to_uint8_image(request[request_key], request_key)
        image_path = output_dir / f"{prefix}_{request_key}.ppm"
        _write_ppm_image(image_path, image)
        written.append(image_path)
        image_metadata[request_key] = {
            "file": image_path.name,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
        }

    metadata_path = output_dir / f"{prefix}_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "call_index": call_index,
                "instruction": request.get("instruction"),
                "state_shape": list(np.asarray(request.get("state")).shape),
                "state_action_order": list(MOLMOACT2_YAM_STATE_ACTION_ORDER),
                "images": image_metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(metadata_path)
    return written


def _as_observation_dict(obs: dict | list | tuple) -> dict:
    if isinstance(obs, list | tuple):
        if len(obs) == 0:
            raise ValueError("Expected at least one observation, got an empty sequence")
        if len(obs) > 1:
            log.warning("Received %d observations; using the first one", len(obs))
        obs = obs[0]
    if not isinstance(obs, dict):
        raise TypeError(f"Expected observation dict, got {type(obs).__name__}")
    return obs


def _to_uint8_image(image: Any, name: str) -> np.ndarray:
    if hasattr(image, "detach") and hasattr(image, "cpu"):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Camera '{name}' must have shape (H, W, 3/4), got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]

    if image.dtype != np.uint8:
        scale = 255.0 if image.size and np.nanmax(image) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return image


def _extract_camera(obs: dict, obs_key: str, request_key: str) -> np.ndarray:
    if obs_key not in obs:
        raise KeyError(
            f"Missing camera '{obs_key}' for MolmoAct2 field '{request_key}'. "
            f"Available observation keys: {sorted(obs.keys())}"
        )
    return _to_uint8_image(obs[obs_key], obs_key)


def _first_scalar(value: Any, name: str) -> float:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()

    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"Expected at least one value for {name}")
    return float(array.reshape(-1)[0])


def _as_float32_vector(value: Any, size: int, name: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()

    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size < size:
        raise ValueError(f"Expected {name} to have at least {size} values, got {vector.size}")
    return vector[:size]


def normalize_yam_gripper_position(
    gripper_qpos: Any,
    gripper_max: float,
    *,
    gripper_open_command: float | None = None,
    gripper_closed_command: float = 0.0,
) -> float:
    """Normalize a YAM gripper position to MolmoAct2 [0, 1], where 1 is open."""

    if gripper_open_command is None:
        gripper_open_command = gripper_max
    span = float(gripper_open_command) - float(gripper_closed_command)
    if abs(span) < 1e-8:
        raise ValueError(
            "gripper_open_command and gripper_closed_command must differ, "
            f"got {gripper_open_command} and {gripper_closed_command}"
        )
    raw = _first_scalar(gripper_qpos, "gripper_qpos")
    return float(np.clip((raw - gripper_closed_command) / span, 0.0, 1.0))


def build_molmoact2_yam_state(
    qpos: dict[str, Any] | np.ndarray,
    *,
    gripper_max: float = MOLMOACT2_YAM_GRIPPER_MAX,
    gripper_open_command: float | None = None,
    gripper_closed_command: float = 0.0,
) -> np.ndarray:
    """Build MolmoAct2 YAM state as ``[L6, L_grip, R6, R_grip]``.

    MolmoSpaces exposes YAM gripper positions in meters with an open command of
    ``0.041`` by default, matching the local YAM MJCF actuator ctrlrange and
    ``BimanualYamGripperGroup``. MolmoAct2 expects the state gripper fields
    normalized to ``[0, 1]`` where ``1`` is open.
    """

    if isinstance(qpos, dict):
        left_arm = _as_float32_vector(qpos.get("left_arm", np.zeros(6)), 6, "left_arm")
        right_arm = _as_float32_vector(qpos.get("right_arm", np.zeros(6)), 6, "right_arm")
        left_gripper = normalize_yam_gripper_position(
            qpos.get("left_gripper", np.array([0.0])),
            gripper_max,
            gripper_open_command=gripper_open_command,
            gripper_closed_command=gripper_closed_command,
        )
        right_gripper = normalize_yam_gripper_position(
            qpos.get("right_gripper", np.array([0.0])),
            gripper_max,
            gripper_open_command=gripper_open_command,
            gripper_closed_command=gripper_closed_command,
        )
        return np.concatenate(
            [
                left_arm,
                np.array([left_gripper], dtype=np.float32),
                right_arm,
                np.array([right_gripper], dtype=np.float32),
            ]
        ).astype(np.float32)

    state = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if state.shape != (MOLMOACT2_YAM_ACTION_DIM,):
        raise ValueError(f"Expected prebuilt YAM state shape (14,), got {state.shape}")
    return state


def build_molmoact2_yam_request(
    obs: dict | list | tuple,
    *,
    instruction: str,
    num_steps: int | None = None,
    gripper_max: float = MOLMOACT2_YAM_GRIPPER_MAX,
    gripper_open_command: float | None = None,
    gripper_closed_command: float = 0.0,
    camera_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a MolmoAct2 YAM ``/act`` request from a MolmoSpaces observation."""

    obs = _as_observation_dict(obs)
    camera_mapping = camera_mapping or DEFAULT_MOLMOACT2_YAM_CAMERA_MAPPING

    qpos = obs["qpos"] if "qpos" in obs else obs.get("state")
    if qpos is None:
        raise KeyError("Observation must include 'qpos' or prebuilt 'state'")

    request: dict[str, Any] = {
        "top_cam": _extract_camera(obs, camera_mapping["top_cam"], "top_cam"),
        "left_cam": _extract_camera(obs, camera_mapping["left_cam"], "left_cam"),
        "right_cam": _extract_camera(obs, camera_mapping["right_cam"], "right_cam"),
        "instruction": str(instruction),
        "state": build_molmoact2_yam_state(
            qpos,
            gripper_max=gripper_max,
            gripper_open_command=gripper_open_command,
            gripper_closed_command=gripper_closed_command,
        ),
    }
    if num_steps is not None:
        request["num_steps"] = int(num_steps)
    return request


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "__numpy__" in value:
        try:
            raw = base64.b64decode(value["__numpy__"])
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(dim) for dim in value["shape"])
        except KeyError as e:
            raise ValueError(f"Malformed json_numpy ndarray missing {e.args[0]!r}") from e
        return np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32, copy=False)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def parse_molmoact2_actions(response: Any) -> np.ndarray:
    """Parse a MolmoAct2 ``/act`` response into a non-empty ``(N, 14)`` array."""

    if isinstance(response, dict):
        if "error" in response:
            raise RuntimeError(f"MolmoAct2 /act returned error: {response['error']}")
        if "actions" not in response:
            raise KeyError("MolmoAct2 /act response dict must include 'actions'")
        response = response["actions"]

    actions = _to_numpy(response)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != MOLMOACT2_YAM_ACTION_DIM:
        raise ValueError(f"Expected MolmoAct2 actions shape (N, 14), got {actions.shape}")
    if actions.shape[0] == 0:
        raise ValueError("MolmoAct2 returned an empty action chunk")
    return actions.astype(np.float32, copy=False)


def scale_molmoact2_yam_gripper_action(
    gripper_value: float,
    *,
    gripper_max: float = MOLMOACT2_YAM_GRIPPER_MAX,
    gripper_open_command: float | None = None,
    gripper_closed_command: float = 0.0,
    grasping_type: str = "continuous",
    grasping_threshold: float = 0.5,
) -> np.ndarray:
    """Scale a MolmoAct2 normalized YAM gripper action to MolmoSpaces meters.

    MolmoSpaces commands YAM gripper opening in meters, with ``gripper_max`` as
    open and ``0`` as closed. Official MolmoAct2 YAM action scalars use
    ``1=open, 0=closed``, so high model values must keep the simulated gripper open.
    """

    if gripper_open_command is None:
        gripper_open_command = gripper_max
    if abs(float(gripper_open_command) - float(gripper_closed_command)) < 1e-8:
        raise ValueError(
            "gripper_open_command and gripper_closed_command must differ, "
            f"got {gripper_open_command} and {gripper_closed_command}"
        )

    if grasping_type == "binary":
        alpha = 1.0 if float(gripper_value) > grasping_threshold else 0.0
    elif grasping_type == "continuous":
        alpha = float(np.clip(float(gripper_value), 0.0, 1.0))
    else:
        raise ValueError(f"Unsupported grasping_type '{grasping_type}'")
    command = float(gripper_closed_command) + alpha * (
        float(gripper_open_command) - float(gripper_closed_command)
    )
    return np.array([command], dtype=np.float32)


def molmoact2_yam_action_to_move_group_command(
    action: Any,
    *,
    gripper_max: float = MOLMOACT2_YAM_GRIPPER_MAX,
    gripper_open_command: float | None = None,
    gripper_closed_command: float = 0.0,
    grasping_type: str = "continuous",
    grasping_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Convert one absolute 14D MolmoAct2 YAM target to MolmoSpaces move groups."""

    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (MOLMOACT2_YAM_ACTION_DIM,):
        raise ValueError(f"Expected MolmoAct2 YAM action shape (14,), got {action.shape}")

    return {
        "left_arm": action[0:6].astype(np.float32),
        "right_arm": action[7:13].astype(np.float32),
        "left_gripper": scale_molmoact2_yam_gripper_action(
            float(action[6]),
            gripper_max=gripper_max,
            gripper_open_command=gripper_open_command,
            gripper_closed_command=gripper_closed_command,
            grasping_type=grasping_type,
            grasping_threshold=grasping_threshold,
        ),
        "right_gripper": scale_molmoact2_yam_gripper_action(
            float(action[13]),
            gripper_max=gripper_max,
            gripper_open_command=gripper_open_command,
            gripper_closed_command=gripper_closed_command,
            grasping_type=grasping_type,
            grasping_threshold=grasping_threshold,
        ),
    }


def _current_yam_command_from_task(task: Any, action: dict[str, Any]) -> dict[str, np.ndarray]:
    robot = task.env.robots[0]
    current = {}
    for move_group_id, target in action.items():
        move_group = robot.robot_view.get_move_group(move_group_id)
        target_shape = np.asarray(target, dtype=np.float32).shape
        joint_pos = np.asarray(move_group.joint_pos, dtype=np.float32).reshape(-1)
        if joint_pos.size == np.prod(target_shape):
            current[move_group_id] = joint_pos.reshape(target_shape)
        elif joint_pos.size > 0 and np.prod(target_shape) == 1:
            current[move_group_id] = joint_pos[:1].reshape(target_shape)
        else:
            current[move_group_id] = np.asarray(move_group.noop_ctrl, dtype=np.float32).reshape(
                target_shape
            )
    return current


def _interpolate_yam_command(
    current: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    *,
    joint_step: float,
    max_smoothing_steps: int,
) -> list[dict[str, np.ndarray]]:
    if joint_step <= 0:
        raise ValueError(f"joint_step must be positive, got {joint_step}")
    if max_smoothing_steps < 1:
        raise ValueError(f"max_smoothing_steps must be >= 1, got {max_smoothing_steps}")

    max_delta = 0.0
    for move_group_id, target_value in target.items():
        target_value = np.asarray(target_value, dtype=np.float32)
        if move_group_id not in current:
            raise KeyError(f"Missing current joint position for move group '{move_group_id}'")
        max_delta = max(max_delta, float(np.abs(current[move_group_id] - target_value).max()))

    steps = min(int(max_delta / joint_step), int(max_smoothing_steps))
    if steps <= 1:
        return [{k: np.asarray(v, dtype=np.float32).copy() for k, v in target.items()}]

    smoothed: list[dict[str, np.ndarray]] = []
    for alpha in np.linspace(0.0, 1.0, steps):
        smoothed.append(
            {
                move_group_id: (
                    current[move_group_id]
                    + alpha * (np.asarray(target_value, dtype=np.float32) - current[move_group_id])
                ).astype(np.float32)
                for move_group_id, target_value in target.items()
            }
        )
    return smoothed


def execute_molmoact2_yam_action(
    task: Any,
    action: dict[str, Any],
    *,
    execution_mode: str = "hardware_smoothing",
    joint_step: float = 0.01,
    max_smoothing_steps: int = 100,
    command_hz: float | None = None,
):
    """Execute one MolmoAct2-YAM action inside MolmoSpaces.

    ``sim_eval_step`` follows the official MolmoAct2 simulation protocol from
    ``sim_eval/run_eval.py``: the policy buffers /act chunks, returns one action
    at a time, and the simulator consumes it with a single environment step.

    ``hardware_smoothing`` is kept only for older MolmoSpaces diagnostics that
    intentionally mirrored the real-robot launcher in
    ``examples/yam/launch_yaml_eval_molmoact.py``.
    """

    if getattr(task.env, "n_batch", 1) != 1:
        raise ValueError("MolmoAct2-YAM execution currently supports n_batch=1 only")

    if execution_mode != "sim_eval_step" and execution_mode != "hardware_smoothing":
        raise ValueError(
            "Unsupported MolmoAct2-YAM execution_mode "
            f"'{execution_mode}'. Expected 'sim_eval_step' or 'hardware_smoothing'."
        )

    if np.all(task.is_done()):
        log.warning("execute_molmoact2_yam_action called on a completed task")
        return task.get_and_cache_all_step_information()

    # Read the initial observation before the first command, matching the
    # generic task.step() first-step check without entering its policy hold.
    if task.num_steps_taken() == 0:
        task.get_observations()

    if isinstance(action, dict) and action.get("done", False):
        action = dict(action)
        action.pop("done")
        task._done_action_received = True

    target = {
        move_group_id: np.asarray(target_value, dtype=np.float32).copy()
        for move_group_id, target_value in action.items()
    }
    if execution_mode == "sim_eval_step":
        command_path = [target]
    else:
        current = _current_yam_command_from_task(task, target)
        command_path = _interpolate_yam_command(
            current,
            target,
            joint_step=joint_step,
            max_smoothing_steps=max_smoothing_steps,
        )

    sim_steps_per_command = task._n_sim_steps_per_ctrl
    if command_hz is not None:
        if command_hz <= 0:
            raise ValueError(f"command_hz must be positive, got {command_hz}")
        ctrl_dt_ms = float(getattr(task, "_ctrl_dt_ms", 0.0))
        if ctrl_dt_ms <= 0:
            raise ValueError("task._ctrl_dt_ms must be positive when command_hz is set")
        sim_dt_ms = ctrl_dt_ms / max(int(task._n_sim_steps_per_ctrl), 1)
        sim_steps_per_command = max(1, int(round((1000.0 / float(command_hz)) / sim_dt_ms)))

    task.episode_step_count += 1

    for command in command_path:
        for robot in task.env.robots:
            robot.update_control(command)
            robot.compute_control()
        task.env.step(sim_steps_per_command)

    task.last_action = action
    observation, reward, terminated, truncated, info = task.get_and_cache_all_step_information()
    done = np.logical_or(terminated, truncated)
    task._cumulative_reward += np.where(done, 0, reward)
    task._num_steps_taken += np.where(done, 0, 1)
    task.action_cache.append(task.last_action)

    return observation, reward, terminated, truncated, info


class MolmoAct2YamPolicy(InferencePolicy):
    """MolmoSpaces policy adapter for remote MolmoAct2-BimanualYAM inference."""

    uses_molmoact2_yam_execution = True

    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
        task=None,
        *,
        client: Any | None = None,
    ) -> None:
        super().__init__(exp_config, task)

        policy_config = exp_config.policy_config
        self.remote_config = getattr(policy_config, "remote_config", None) or {}
        self.endpoint_url = (
            getattr(policy_config, "endpoint_url", None)
            or self.remote_config.get("endpoint_url")
            or self.remote_config.get("url")
        )
        self.request_timeout = float(
            getattr(
                policy_config,
                "request_timeout",
                getattr(
                    policy_config,
                    "timeout",
                    self.remote_config.get(
                        "request_timeout", self.remote_config.get("timeout", 60.0)
                    ),
                ),
            )
        )
        self.num_steps = getattr(
            policy_config, "num_steps", self.remote_config.get("num_steps", None)
        )
        self.n_action_steps = getattr(
            policy_config,
            "n_action_steps",
            getattr(policy_config, "action_horizon", None),
        )
        self.gripper_max = float(getattr(policy_config, "gripper_max", MOLMOACT2_YAM_GRIPPER_MAX))
        configured_open_command = getattr(policy_config, "gripper_open_command", None)
        self.gripper_open_command = (
            None if configured_open_command is None else float(configured_open_command)
        )
        self.gripper_closed_command = float(getattr(policy_config, "gripper_closed_command", 0.0))
        self.gripper_scale_source = getattr(
            policy_config,
            "gripper_scale_source",
            MOLMOACT2_YAM_GRIPPER_SCALE_SOURCE,
        )
        self.grasping_type = getattr(policy_config, "grasping_type", "continuous")
        self.grasping_threshold = float(getattr(policy_config, "grasping_threshold", 0.5))
        self.instruction_override = getattr(
            policy_config,
            "instruction_override",
            DEFAULT_MOLMOACT2_YAM_INSTRUCTION,
        )
        self.debug_dump_dir = getattr(policy_config, "debug_dump_dir", None)
        self.debug_dump_max_calls = int(getattr(policy_config, "debug_dump_max_calls", 0) or 0)
        self.raw_action_log_path = getattr(policy_config, "raw_action_log_path", None)
        self.execution_mode = getattr(policy_config, "execution_mode", "hardware_smoothing")
        self.execution_joint_step = float(getattr(policy_config, "execution_joint_step", 0.01))
        self.execution_max_smoothing_steps = int(
            getattr(policy_config, "execution_max_smoothing_steps", 100)
        )
        execution_command_hz = getattr(policy_config, "execution_command_hz", None)
        self.execution_command_hz = (
            None if execution_command_hz is None else float(execution_command_hz)
        )
        self.camera_mapping = dict(
            getattr(
                policy_config,
                "camera_mapping",
                DEFAULT_MOLMOACT2_YAM_CAMERA_MAPPING,
            )
        )

        self.client = client
        self.actions_buffer: list[np.ndarray] | None = None
        self.current_buffer_index = 0
        self.inference_call_count = 0
        self.starting_time: float | None = None
        self.model_name = "molmoact2_yam"

    def reset(self) -> None:
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.inference_call_count = 0
        self.starting_time = None
        if self.client is not None and hasattr(self.client, "reset"):
            self.client.reset()

    def prepare_model(self) -> None:
        if self.client is not None:
            return

        endpoint_url = self.endpoint_url
        if endpoint_url is None:
            host = self.remote_config.get("host")
            if host is not None:
                port = self.remote_config.get("port", 8202)
                path = self.remote_config.get("path", "/act")
                endpoint_url = f"http://{host}:{port}{path}"

        if endpoint_url is None:
            raise ValueError(
                "MolmoAct2YamPolicy requires an endpoint_url/remote_config host "
                "or an injected fake client"
            )

        self.endpoint_url = endpoint_url
        self.client = MolmoAct2HTTPClient(endpoint_url, request_timeout=self.request_timeout)
        log.info("MolmoAct2YamPolicy connected to %s", endpoint_url)

    def obs_to_model_input(self, obs: dict) -> dict[str, Any]:
        return build_molmoact2_yam_request(
            obs,
            instruction=self._get_instruction(obs),
            num_steps=self.num_steps,
            gripper_max=self.gripper_max,
            gripper_open_command=self.gripper_open_command,
            gripper_closed_command=self.gripper_closed_command,
            camera_mapping=self.camera_mapping,
        )

    def _get_instruction(self, obs: dict) -> str:
        obs_dict = _as_observation_dict(obs)
        if self.instruction_override:
            return str(self.instruction_override)
        if self.task is not None and hasattr(self.task, "get_task_description"):
            return self.task.get_task_description()
        if self.task is not None and isinstance(self.task, str):
            return self.task
        if "instruction" in obs_dict:
            return str(obs_dict["instruction"])
        if "task" in obs_dict:
            return str(obs_dict["task"])
        return ""

    def inference_model(self, model_input: dict[str, Any]) -> np.ndarray:
        if self.starting_time is None:
            self.starting_time = time.time()

        if self.actions_buffer is None or self.current_buffer_index >= len(self.actions_buffer):
            if self.client is None:
                self.prepare_model()

            call_index = self.inference_call_count
            self.inference_call_count += 1
            if self.debug_dump_dir and call_index < self.debug_dump_max_calls:
                written = dump_molmoact2_yam_request_debug_images(
                    model_input,
                    self.debug_dump_dir,
                    call_index=call_index,
                )
                log.info(
                    "Dumped MolmoAct2 YAM /act debug images for call %d to %s (%d files)",
                    call_index,
                    self.debug_dump_dir,
                    len(written),
                )

            response = self._call_client(model_input)
            actions = parse_molmoact2_actions(response)
            if self.raw_action_log_path:
                log_path = append_molmoact2_yam_raw_action_log(
                    self.raw_action_log_path,
                    call_index=call_index,
                    request=model_input,
                    response=response,
                    actions=actions,
                    camera_mapping=self.camera_mapping,
                )
                log.info(
                    "Appended MolmoAct2 YAM raw action log for call %d to %s", call_index, log_path
                )
            if self.n_action_steps is not None:
                actions = actions[: max(1, int(self.n_action_steps))]

            self.actions_buffer = [np.asarray(action, dtype=np.float32) for action in actions]
            self.current_buffer_index = 0

        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def _call_client(self, model_input: dict[str, Any]) -> Any:
        if hasattr(self.client, "act"):
            return self.client.act(model_input)
        if callable(self.client):
            return self.client(model_input)
        raise TypeError("MolmoAct2 client must provide act(request) or be callable")

    def model_output_to_action(self, model_output: np.ndarray) -> dict[str, np.ndarray]:
        model_output = np.asarray(model_output, dtype=np.float32).reshape(-1)
        action = self.raw_action_to_move_group_command(
            model_output,
            gripper_max=self.gripper_max,
            gripper_open_command=self.gripper_open_command,
            gripper_closed_command=self.gripper_closed_command,
            grasping_type=self.grasping_type,
            grasping_threshold=self.grasping_threshold,
        )

        log.debug(
            "MolmoAct2 action step=%d left_grip=%.3f right_grip=%.3f",
            self.current_buffer_index,
            model_output[6],
            model_output[13],
        )
        return action

    @staticmethod
    def raw_action_to_move_group_command(
        model_output: Any,
        *,
        gripper_max: float = MOLMOACT2_YAM_GRIPPER_MAX,
        gripper_open_command: float | None = None,
        gripper_closed_command: float = 0.0,
        grasping_type: str = "continuous",
        grasping_threshold: float = 0.5,
    ) -> dict[str, np.ndarray]:
        return molmoact2_yam_action_to_move_group_command(
            model_output,
            gripper_max=gripper_max,
            gripper_open_command=gripper_open_command,
            gripper_closed_command=gripper_closed_command,
            grasping_type=grasping_type,
            grasping_threshold=grasping_threshold,
        )

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "molmoact2_yam"
        info["policy_endpoint_url"] = self.endpoint_url
        info["policy_num_steps"] = self.num_steps
        info["policy_n_action_steps"] = self.n_action_steps
        info["policy_gripper_max"] = self.gripper_max
        info["policy_gripper_open_command"] = self.gripper_open_command
        info["policy_gripper_closed_command"] = self.gripper_closed_command
        info["policy_gripper_scale_source"] = self.gripper_scale_source
        info["policy_action_gripper_semantics"] = MOLMOACT2_YAM_ACTION_GRIPPER_SEMANTICS
        info["policy_grasping_type"] = self.grasping_type
        info["policy_instruction_override"] = self.instruction_override
        info["policy_debug_dump_dir"] = self.debug_dump_dir
        info["policy_debug_dump_max_calls"] = self.debug_dump_max_calls
        info["policy_raw_action_log_path"] = self.raw_action_log_path
        info["policy_execution_mode"] = self.execution_mode
        info["policy_execution_command_hz"] = self.execution_command_hz
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        return info

    def __del__(self):
        if hasattr(self, "client") and self.client is not None and hasattr(self.client, "close"):
            self.client.close()
