"""Gate 1D-2C: official-aligned browser keyboard teleop for bimanual YAM.

The policy math and lifecycle mirror MolmoSpaces Keyboard_Policy:
- InferencePolicy.get_action(observation)
- robot-relative TCP target state
- current_position += current_rotation @ delta_position
- current_rotation = R.from_euler("xyz", delta_rotation) @ current_rotation
- official TeleopPolicyConfig defaults: step_size=0.005 m, rot_step=0.02 rad

Only the input/display transport and the minimal single-arm -> active-arm generalization are new.
No browser reset/save controls are exposed in this gate.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import signal
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from molmo_spaces.env.sensors_cameras import CameraSensor as _CameraSensor
from molmo_spaces.policy.base_policy import InferencePolicy

from check_dual_object_reachability import (
    CAMERAS,
    to_jsonable,
)
from validate_tabletop_initialization import (
    sample_strict_tabletop_task,
)

ARMS = ("left", "right")
GRIPPER_MAX = 0.041
OFFICIAL_STEP_SIZE = 0.005
OFFICIAL_ROT_STEP = 0.02
HEADER = struct.Struct(">IdI")  # sequence, captured_at, status_json_length

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MolmoSpaces official-aligned bimanual teleop</title>
<style>
body{margin:0;background:#151719;color:#eee;font:14px system-ui,sans-serif}header{padding:9px 14px;background:#202326;display:flex;gap:14px;align-items:center;flex-wrap:wrap}b,.ok{color:#70d6a8}.warn{color:#ffd166}.bad{color:#ff8d8d}main{padding:12px}img{display:block;width:100%;max-width:1280px;background:#000;border:1px solid #444}.help{margin-top:10px;line-height:1.8;color:#cfd4d7}kbd{background:#34383c;border:1px solid #666;border-radius:4px;padding:1px 6px}.pill{padding:2px 7px;border:1px solid #555;border-radius:10px}
</style></head><body><header><strong>MolmoSpaces official-aligned teleop</strong><span id="conn">connecting</span><span id="arm" class="pill">active: left</span><span id="grip">L open · R open</span><span id="fps"></span><span id="age"></span><span id="safe" class="warn">safe hold</span><span id="ik"></span><span id="err" class="bad"></span></header><main><img id="view" alt="top, left wrist, right wrist"><div class="help"><b>Click the page first.</b><br>
<kbd>Tab</kbd> switch arm · <kbd>W/S</kbd> visual forward/back · <kbd>A/D</kbd> visual left/right · <kbd>E/Q</kbd> raise/lower · <kbd>←/→</kbd> yaw +/− · <kbd>↑/↓</kbd> pitch +/− · <kbd>Z/C</kbd> roll +/− · <kbd>F</kbd> active gripper.<br>
Position/rotation update, IK lifecycle, 5 mm step and 0.02 rad rotation match MolmoSpaces Keyboard_Policy. Blur, hidden tab, disconnect, or 400 ms timeout releases all motion. No reset/save in this gate.</div></main>
<script>
const $=s=>document.querySelector(s),conn=$('#conn'),armEl=$('#arm'),grip=$('#grip'),fps=$('#fps'),age=$('#age'),safe=$('#safe'),ik=$('#ik'),err=$('#err'),view=$('#view');
let ws=null,seq=0,active='left',keys=new Set(),grippers={left:true,right:true},lastFrame=0,lastFrameSeq=-1,frames=0,windowAt=performance.now();
const controlled=new Set(['KeyW','KeyS','KeyA','KeyD','KeyQ','KeyE','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','KeyZ','KeyC']);
function motion(){return {navigation:[(keys.has('KeyW')?1:0)-(keys.has('KeyS')?1:0),(keys.has('KeyD')?1:0)-(keys.has('KeyA')?1:0),(keys.has('KeyE')?1:0)-(keys.has('KeyQ')?1:0)],rotation:[(keys.has('KeyZ')?1:0)-(keys.has('KeyC')?1:0),(keys.has('ArrowUp')?1:0)-(keys.has('ArrowDown')?1:0),(keys.has('ArrowLeft')?1:0)-(keys.has('ArrowRight')?1:0)]}}
function send(release=false){if(!ws||ws.readyState!==1)return;const m=motion();ws.send(JSON.stringify({type:'control',sequence:++seq,active_arm:active,navigation:release?[0,0,0]:m.navigation,rotation:release?[0,0,0]:m.rotation,gripper_open:grippers,release_all:release}));}
function releaseAll(){keys.clear();send(true)}
addEventListener('keydown',e=>{if(e.code==='Tab'){e.preventDefault();if(!e.repeat){active=active==='left'?'right':'left';armEl.textContent=`active: ${active}`;releaseAll()};return}if(e.code==='KeyF'){e.preventDefault();if(!e.repeat){grippers[active]=!grippers[active];send()};return}if(controlled.has(e.code)){e.preventDefault();keys.add(e.code);send()}});
addEventListener('keyup',e=>{if(controlled.has(e.code)){e.preventDefault();keys.delete(e.code);send()}});addEventListener('blur',releaseAll);document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseAll()});setInterval(()=>send(false),80);
function connect(){ws=new WebSocket(`ws://${location.host}/stream`);ws.binaryType='arraybuffer';ws.onopen=()=>{conn.innerHTML='<b>connected</b>';err.textContent='';seq=0;releaseAll()};
ws.onmessage=e=>{if(typeof e.data==='string')return;const data=e.data,h=new DataView(data,0,16),statusLen=h.getUint32(12),status=JSON.parse(new TextDecoder().decode(data.slice(16,16+statusLen)));lastFrameSeq=h.getUint32(0);lastFrame=performance.now();safe.textContent=status.input_fresh?'input live':'safe hold';safe.className=status.input_fresh?'ok':'warn';ik.textContent=`IK ${status.ik_status}, step ${status.step}`;grip.textContent=`L ${status.gripper_open.left?'open':'closed'} · R ${status.gripper_open.right?'open':'closed'}`;const blob=new Blob([data.slice(16+statusLen)],{type:'image/jpeg'}),old=view.src;view.src=URL.createObjectURL(blob);if(old)URL.revokeObjectURL(old);frames++;const now=performance.now();if(now-windowAt>=1000){fps.textContent=`${(frames*1000/(now-windowAt)).toFixed(1)} FPS`;frames=0;windowAt=now}};
ws.onclose=()=>{conn.textContent='disconnected';keys.clear();setTimeout(connect,1000)};ws.onerror=()=>{err.textContent='socket error'}}connect();setInterval(()=>{if(lastFrame)age.textContent=`frame ${lastFrameSeq} · age ${Math.round(performance.now()-lastFrame)} ms`},500);
</script></body></html>"""


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError(f"degenerate {name}")
    return vector / norm


def visual_basis(camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return visual forward, right, world-up axes in world coordinates.

    Visual forward is the image-up direction projected onto the horizontal plane; visual
    right is the renderer's right direction projected horizontally. This makes W move
    toward the top of the exocentric image and D toward its right edge.
    """
    world_up = np.array([0.0, 0.0, 1.0])
    forward = _unit(camera.forward, "camera forward")
    up = _unit(camera.up, "camera up")
    right = _unit(np.cross(forward, up), "camera right")
    visual_forward = up - world_up * float(np.dot(up, world_up))
    if np.linalg.norm(visual_forward) < 1e-6:
        optical_horizontal = forward - world_up * float(np.dot(forward, world_up))
        visual_forward = optical_horizontal
    visual_forward = _unit(visual_forward, "visual forward")
    visual_right = right - world_up * float(np.dot(right, world_up))
    visual_right = _unit(visual_right, "visual right")
    if float(np.dot(np.cross(visual_right, visual_forward), world_up)) < 0:
        visual_right *= -1.0
    return visual_forward, visual_right, world_up


@dataclass
class SharedFrame:
    condition: threading.Condition = field(default_factory=threading.Condition)
    sequence: int = 0
    captured_at: float = 0.0
    payload: bytes = b""
    stopped: bool = False

    def publish(self, jpeg: bytes, status: dict[str, Any]) -> None:
        status_bytes = json.dumps(to_jsonable(status), separators=(",", ":")).encode("utf-8")
        with self.condition:
            self.sequence += 1
            self.captured_at = time.time()
            self.payload = (
                HEADER.pack(self.sequence, self.captured_at, len(status_bytes))
                + status_bytes
                + jpeg
            )
            self.condition.notify_all()

    def snapshot_after(self, sequence: int, timeout: float = 1.0):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > sequence or self.stopped, timeout)
            return self.sequence, self.payload, self.stopped

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()


@dataclass
class ControlSnapshot:
    sequence: int
    active_arm: str
    navigation: np.ndarray  # visual forward, visual right, world up
    rotation: np.ndarray  # official xyz Euler: roll, pitch, yaw
    gripper_open: dict[str, bool]
    input_fresh: bool


@dataclass
class SharedControl:
    lock: threading.Lock = field(default_factory=threading.Lock)
    sequence: int = -1
    active_arm: str = "left"
    navigation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    gripper_open: dict[str, bool] = field(default_factory=lambda: {"left": True, "right": True})
    received_at: float = -math.inf
    connected: bool = False
    rejected_out_of_order: int = 0
    rejected_invalid: int = 0

    def begin_session(self) -> None:
        with self.lock:
            self.sequence = -1
            self.navigation[:] = 0.0
            self.rotation[:] = 0.0
            self.received_at = -math.inf
            self.connected = True

    def disconnect(self) -> None:
        with self.lock:
            self.navigation[:] = 0.0
            self.rotation[:] = 0.0
            self.received_at = -math.inf
            self.connected = False

    def apply(self, payload: dict[str, Any], now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        try:
            if payload.get("type") != "control":
                raise ValueError("unknown message type")
            sequence = int(payload["sequence"])
            active_arm = str(payload["active_arm"])
            navigation = np.asarray(payload["navigation"], dtype=float)
            rotation = np.asarray(payload["rotation"], dtype=float)
            gripper_open = payload["gripper_open"]
            if active_arm not in ARMS or navigation.shape != (3,) or rotation.shape != (3,):
                raise ValueError("invalid arm or shape")
            if not np.isfinite(navigation).all() or not np.isfinite(rotation).all():
                raise ValueError("non-finite control")
            if np.max(np.abs(navigation)) > 1.0 or np.max(np.abs(rotation)) > 1.0:
                raise ValueError("control outside [-1,1]")
            if set(gripper_open) != set(ARMS) or not all(
                isinstance(gripper_open[x], bool) for x in ARMS
            ):
                raise ValueError("invalid gripper state")
        except (KeyError, TypeError, ValueError, OverflowError):
            with self.lock:
                self.rejected_invalid += 1
            return False
        with self.lock:
            if sequence <= self.sequence:
                self.rejected_out_of_order += 1
                return False
            self.sequence = sequence
            self.active_arm = active_arm
            release = bool(payload.get("release_all", False))
            self.navigation = np.zeros(3) if release else navigation.copy()
            self.rotation = np.zeros(3) if release else rotation.copy()
            self.gripper_open = {side: bool(gripper_open[side]) for side in ARMS}
            self.received_at = now
            return True

    def snapshot(self, timeout_s: float, now: float | None = None) -> ControlSnapshot:
        now = time.monotonic() if now is None else now
        with self.lock:
            fresh = self.connected and now - self.received_at <= timeout_s
            return ControlSnapshot(
                self.sequence,
                self.active_arm,
                self.navigation.copy() if fresh else np.zeros(3),
                self.rotation.copy() if fresh else np.zeros(3),
                dict(self.gripper_open),
                fresh,
            )


class BrowserBimanualKeyboardPolicy(InferencePolicy):
    """Minimal bimanual generalization of official MolmoSpaces Keyboard_Policy."""

    def __init__(
        self,
        config,
        task,
        shared_control: SharedControl,
        timeout_s: float,
        step_size: float,
        rot_step: float,
    ):
        super().__init__(config, task)
        self.shared_control = shared_control
        self.timeout_s = timeout_s
        self.step_size = step_size
        self.rot_step = rot_step
        self.current_position: dict[str, np.ndarray] = {}
        self.current_rotation: dict[str, np.ndarray] = {}
        self.last_snapshot = ControlSnapshot(
            -1, "left", np.zeros(3), np.zeros(3), {"left": True, "right": True}, False
        )
        self.ik_status = "hold"
        self.last_policy_latency_ms = 0.0
        self.last_axes: dict[str, list[float]] = {}

    def prepare_model(self, model_name: str | None = None):
        return None

    def reset(self):
        self.current_position.clear()
        self.current_rotation.clear()
        self.ik_status = "hold"

    def obs_to_model_input(self, obs):
        if isinstance(obs, list):
            obs = obs[0]
        robot_pose = np.asarray(obs["robot_base_pose"], dtype=float)
        t_world_robot = np.eye(4)
        t_world_robot[:3, :3] = R.from_quat(robot_pose[3:], scalar_first=True).as_matrix()
        t_world_robot[:3, 3] = robot_pose[:3]
        tcp = {}
        for side in ARMS:
            pose = np.asarray(obs[f"tcp_pose_{side}"], dtype=float)
            matrix = np.eye(4)
            matrix[:3, :3] = R.from_quat(pose[3:], scalar_first=True).as_matrix()
            matrix[:3, 3] = pose[:3]
            tcp[side] = matrix
            if side not in self.current_position:
                self.current_position[side] = matrix[:3, 3].copy()
                self.current_rotation[side] = matrix[:3, :3].copy()
        self.task.env.camera_manager.registry.update_all_cameras(self.task.env)
        camera = self.task.env.camera_manager.registry["exo_camera"]
        visual_forward_world, visual_right_world, world_up = visual_basis(camera)
        self.last_axes = {
            "visual_forward_world": visual_forward_world.tolist(),
            "visual_right_world": visual_right_world.tolist(),
            "world_up": world_up.tolist(),
        }
        return {
            "qpos": {k: np.asarray(v).copy() for k, v in obs["qpos"].items()},
            "T_world_robot": t_world_robot,
            "tcp": tcp,
            "visual_axes_world": (visual_forward_world, visual_right_world, world_up),
        }

    def inference_model(self, model_input):
        started = time.perf_counter()
        snapshot = self.shared_control.snapshot(self.timeout_s)
        self.last_snapshot = snapshot
        qpos = model_input["qpos"]
        action = {
            "left_arm": qpos["left_arm"].copy(),
            "right_arm": qpos["right_arm"].copy(),
            "left_gripper": np.array([GRIPPER_MAX if snapshot.gripper_open["left"] else 0.0]),
            "right_gripper": np.array([GRIPPER_MAX if snapshot.gripper_open["right"] else 0.0]),
        }
        if not snapshot.input_fresh or (
            not np.any(snapshot.navigation) and not np.any(snapshot.rotation)
        ):
            self.ik_status = "hold"
            self.last_policy_latency_ms = (time.perf_counter() - started) * 1000.0
            return action

        side = snapshot.active_arm
        arm, gripper = f"{side}_arm", f"{side}_gripper"
        vf_world, vr_world, up_world = model_input["visual_axes_world"]
        desired_world = (
            snapshot.navigation[0] * vf_world
            + snapshot.navigation[1] * vr_world
            + snapshot.navigation[2] * up_world
        )
        desired_robot = model_input["T_world_robot"][:3, :3].T @ desired_world

        # Exact official Keyboard_Policy position/rotation update form.
        local_delta = self.current_rotation[side].T @ (desired_robot * self.step_size)
        candidate_position = self.current_position[side] + self.current_rotation[side] @ local_delta
        delta_rotation = R.from_euler("xyz", snapshot.rotation * self.rot_step).as_matrix()
        candidate_rotation = delta_rotation @ self.current_rotation[side]
        new_pose = np.eye(4)
        new_pose[:3, 3] = candidate_position
        new_pose[:3, :3] = candidate_rotation

        robot = self.task.env.current_robot
        view = robot.robot_view
        solution = robot.kinematics.ik(
            gripper,
            new_pose,
            [arm],
            view.get_qpos_dict(),
            view.base.pose,
            rel_to_base=True,
        )
        if solution is None or arm not in solution or not np.isfinite(solution[arm]).all():
            self.ik_status = "failed_hold"
            self.last_policy_latency_ms = (time.perf_counter() - started) * 1000.0
            return action
        self.current_position[side] = candidate_position
        self.current_rotation[side] = candidate_rotation
        action[arm] = np.asarray(solution[arm]).copy()
        self.ik_status = "success"
        self.last_policy_latency_ms = (time.perf_counter() - started) * 1000.0
        return action

    def model_output_to_action(self, model_output):
        return model_output

    def get_info(self) -> dict:
        info = super().get_info()
        info.update(
            {
                "policy_name": "browser_bimanual_keyboard_official_aligned",
                "step_size": self.step_size,
                "rot_step": self.rot_step,
            }
        )
        return info


def compose_frame(observation, width: int, height: int, quality: int) -> bytes:
    """Encode the RGB frames already produced by task.step().

    The official sensor suite renders all configured cameras while constructing the
    returned observation. Re-rendering them here would duplicate the dominant work
    and reduce interactive throughput without changing the displayed state.
    """
    obs = observation[0] if isinstance(observation, list) else observation
    panels = []
    for camera_name in CAMERAS:
        frame = obs[camera_name]
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize(
            (width, height), Image.Resampling.BILINEAR
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 190, 24), fill=(0, 0, 0))
        draw.text((7, 5), camera_name, fill=(255, 255, 255))
        panels.append(image)
    composite = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        composite.paste(panel, (index * width, 0))
    buffer = io.BytesIO()
    composite.save(buffer, format="JPEG", quality=quality, optimize=False)
    return buffer.getvalue()


def _pop_camera_sensors(task) -> dict:
    """Temporarily remove CameraSensor instances from sensor_suite; return them for later restore."""
    removed: dict = {}
    if task.sensor_suite is None:
        return removed
    for k in list(task.sensor_suite.sensors.keys()):
        if isinstance(task.sensor_suite.sensors[k], _CameraSensor):
            removed[k] = task.sensor_suite.sensors.pop(k)
    return removed


def _restore_camera_sensors(task, removed: dict) -> None:
    """Re-insert previously removed camera sensors into sensor_suite."""
    if task.sensor_suite is not None:
        task.sensor_suite.sensors.update(removed)


def trim_task_caches(task, keep: int = 2) -> None:
    for name in (
        "observation_cache",
        "reward_cache",
        "terminal_cache",
        "truncated_cache",
        "success_cache",
        "action_cache",
    ):
        cache = getattr(task, name, None)
        if isinstance(cache, list) and len(cache) > keep:
            del cache[:-keep]


async def run_server(
    shared_frame: SharedFrame, shared_control: SharedControl, host: str, port: int
) -> None:
    async def process_request(connection: ServerConnection, request):
        if request.path == "/healthz":
            return connection.respond(200, "OK\n")
        if request.path == "/":
            response = connection.respond(200, HTML)
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response
        if request.path == "/favicon.ico":
            return connection.respond(204, "")
        if request.path == "/stream":
            return None
        return connection.respond(404, "Not Found\n")

    async def handler(websocket: ServerConnection) -> None:
        if websocket.request.path != "/stream":
            await websocket.close(code=1008, reason="unknown websocket path")
            return
        shared_control.begin_session()

        async def receive_controls() -> None:
            async for message in websocket:
                if not isinstance(message, str):
                    continue
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    with shared_control.lock:
                        shared_control.rejected_invalid += 1
                    continue
                shared_control.apply(payload)

        async def send_frames() -> None:
            sequence = -1
            while True:
                sequence, payload, stopped = await asyncio.to_thread(
                    shared_frame.snapshot_after, sequence
                )
                if stopped:
                    return
                if payload:
                    await websocket.send(payload)  # exactly one message per frame

        try:
            await asyncio.gather(receive_controls(), send_frames())
        except ConnectionClosed:
            pass
        finally:
            shared_control.disconnect()

    async with serve(
        handler,
        host,
        port,
        compression=None,
        max_size=1_000_000,
        process_request=process_request,
        ping_interval=20,
        ping_timeout=20,
    ) as server:
        print(f"teleop=http://{host}:{port}", flush=True)
        await server.serve_forever()


def exercise_protocol_logic() -> dict[str, Any]:
    shared = SharedControl()
    shared.begin_session()
    valid = {
        "type": "control",
        "sequence": 1,
        "active_arm": "left",
        "navigation": [1, 0, 0],
        "rotation": [0, 0, 0],
        "gripper_open": {"left": True, "right": True},
        "release_all": False,
    }
    accepted = shared.apply(valid, now=10.0)
    fresh = shared.snapshot(0.4, now=10.2)
    stale = shared.snapshot(0.4, now=10.5)
    out_of_order = shared.apply({**valid, "sequence": 1}, now=10.6)
    invalid = shared.apply({**valid, "sequence": 2, "rotation": [0, 2, 0]}, now=10.7)
    release = shared.apply({**valid, "sequence": 3, "release_all": True}, now=10.8)
    released = shared.snapshot(0.4, now=10.9)
    result = {
        "accepted": accepted,
        "fresh_nonzero": bool(fresh.input_fresh and np.linalg.norm(fresh.navigation) > 0),
        "stale_zero": bool(
            not stale.input_fresh
            and np.allclose(stale.navigation, 0)
            and np.allclose(stale.rotation, 0)
        ),
        "out_of_order_rejected": not out_of_order,
        "invalid_rejected": not invalid,
        "release_zero": bool(
            release and np.allclose(released.navigation, 0) and np.allclose(released.rotation, 0)
        ),
    }
    result["pass"] = all(result.values())
    return result


def exercise_visual_mapping() -> dict[str, Any]:
    class Camera:
        forward = np.array([0.0, 0.0, -1.0])
        up = np.array([1.0, 0.0, 0.0])

    vf, vr, wu = visual_basis(Camera())
    checks = {
        "w_is_image_up": bool(np.dot(vf, Camera.up) > 0.999),
        "s_is_opposite_w": bool(np.allclose(-vf, -Camera.up)),
        "d_is_image_right": bool(np.linalg.norm(vr) == 1.0),
        "e_is_world_up": bool(np.allclose(wu, [0, 0, 1])),
        "official_step_size": OFFICIAL_STEP_SIZE,
        "official_rot_step": OFFICIAL_ROT_STEP,
    }
    checks["pass"] = all(
        checks[k] for k in ("w_is_image_up", "s_is_opposite_w", "d_is_image_right", "e_is_world_up")
    )
    return checks


def run_sim_smoke(
    task, env, config, timeout_s: float, step_size: float, rot_step: float
) -> dict[str, Any]:
    shared = SharedControl()
    shared.begin_session()
    policy = BrowserBimanualKeyboardPolicy(config, task, shared, timeout_s, step_size, rot_step)
    task.register_policy(policy)
    observation, _ = task.reset()
    mujoco.mj_forward(env.current_model, env.current_data)
    base = {
        "type": "control",
        "active_arm": "left",
        "gripper_open": {"left": True, "right": True},
        "release_all": False,
    }
    checks = {
        "protocol": exercise_protocol_logic(),
        "mapping": exercise_visual_mapping(),
        "commands": {},
    }
    commands = {
        "forward": ([1, 0, 0], [0, 0, 0]),
        "back": ([-1, 0, 0], [0, 0, 0]),
        "left": ([0, -1, 0], [0, 0, 0]),
        "right": ([0, 1, 0], [0, 0, 0]),
        "raise": ([0, 0, 1], [0, 0, 0]),
        "lower": ([0, 0, -1], [0, 0, 0]),
        "roll": ([0, 0, 0], [1, 0, 0]),
        "pitch": ([0, 0, 0], [0, 1, 0]),
        "yaw": ([0, 0, 0], [0, 0, 1]),
    }
    sequence = 0
    for side in ARMS:
        checks["commands"][side] = {}
        for name, (navigation, rotation) in commands.items():
            sequence += 1
            shared.begin_session()
            policy.reset()
            shared.apply(
                {
                    **base,
                    "sequence": sequence,
                    "active_arm": side,
                    "navigation": navigation,
                    "rotation": rotation,
                },
                now=time.monotonic(),
            )
            q_before = {
                k: v.copy() for k, v in env.current_robot.robot_view.get_qpos_dict().items()
            }
            action = policy.get_action(observation)
            inactive = "right" if side == "left" else "left"
            finite = all(np.isfinite(v).all() for v in action.values())
            inactive_held = np.allclose(action[f"{inactive}_arm"], q_before[f"{inactive}_arm"])
            ik_status = policy.ik_status
            checks["commands"][side][name] = {
                "ik_status": ik_status,
                "policy_latency_ms": policy.last_policy_latency_ms,
                "finite": finite,
                "inactive_held": bool(inactive_held),
                "pass": bool(ik_status == "success" and finite and inactive_held),
            }
            shared.disconnect()
    command_pass = all(
        item["pass"]
        for side_results in checks["commands"].values()
        for item in side_results.values()
    )
    checks["strict_pass"] = bool(
        checks["protocol"]["pass"] and checks["mapping"]["pass"] and command_pass
    )
    checks["official_alignment"] = {
        "base_class": "InferencePolicy",
        "lifecycle": "policy.get_action(observation) -> task.step(action)",
        "step_size": step_size,
        "rot_step": rot_step,
        "ik_defaults": "official MlSpacesKinematics.ik defaults",
        "position_update": "current_position += current_rotation @ delta_position",
        "rotation_update": "R.from_euler('xyz', delta) @ current_rotation",
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--house-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--control-hz", type=float, default=25.0)
    parser.add_argument("--render-fps", type=float, default=8.0)
    parser.add_argument("--input-timeout-ms", type=float, default=400.0)
    parser.add_argument("--step-size", type=float, default=OFFICIAL_STEP_SIZE)
    parser.add_argument("--rot-step", type=float, default=OFFICIAL_ROT_STEP)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--initialization-report", type=Path)
    parser.add_argument("--initialization-max-attempts", type=int, default=8)
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("loopback only")
    if args.step_size != OFFICIAL_STEP_SIZE or args.rot_step != OFFICIAL_ROT_STEP:
        raise ValueError("Gate 1D-2C requires official step_size=0.005 and rot_step=0.02")
    shared_frame = SharedFrame()
    shared_control = SharedControl()
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    sampler = None
    try:
        sampler, task, attempt_records = sample_strict_tabletop_task(
            args.house_index,
            args.seed,
            args.initialization_max_attempts,
            policy_dt_ms=40,
            task_horizon=100000,
        )
        config = sampler.config
        env = sampler.env
        initialization = sampler.accepted_initialization
        if not initialization or not initialization.get("strict_pass", False):
            raise RuntimeError("tabletop initialization did not pass strict acceptance gate")
        if args.initialization_report:
            args.initialization_report.parent.mkdir(parents=True, exist_ok=True)
            args.initialization_report.write_text(
                json.dumps(
                    to_jsonable(
                        {
                            "gate": "1D-2C-tabletop-initialization",
                            "house_index": args.house_index,
                            "base_seed": args.seed,
                            "accepted": initialization,
                            "attempt_records": attempt_records,
                            "evidence_boundary": "Official FloorPlan1 geometry and official placement flow with additive island/bimanual acceptance gate; not an untouched upstream preset, human demo, grasp, packing success, or replay.",
                        }
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        if args.smoke:
            result = run_sim_smoke(
                task, env, config, args.input_timeout_ms / 1000, args.step_size, args.rot_step
            )
            report = {
                "gate": "1D-2C-official-aligned",
                "house_index": args.house_index,
                "seed": args.seed,
                "result": result,
                "evidence_boundary": "Official-aligned control technical smoke; not operator visual confirmation, grasp, packing, saved demo, or replay.",
            }
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(
                    json.dumps(to_jsonable(report), indent=2, ensure_ascii=False) + "\n"
                )
            print(json.dumps(to_jsonable(report), indent=2, ensure_ascii=False))
            if not result["strict_pass"]:
                raise SystemExit(2)
            return
        policy = BrowserBimanualKeyboardPolicy(
            config,
            task,
            shared_control,
            args.input_timeout_ms / 1000,
            args.step_size,
            args.rot_step,
        )
        task.register_policy(policy)
        observation, _ = task.reset()
        mujoco.mj_forward(env.current_model, env.current_data)
        thread = threading.Thread(
            target=lambda: asyncio.run(
                run_server(shared_frame, shared_control, args.host, args.port)
            ),
            daemon=True,
        )
        thread.start()
        started = time.monotonic()
        period = 1 / args.control_hz
        next_render = 0.0
        step = 0
        frames_sent = 0
        server_fps_elapsed = 0.0
        server_fps_window = started
        _last_cam_frames: dict = {}  # stale camera frames reused on non-render cycles
        while not stop_event.is_set():
            cycle = time.monotonic()
            do_render = cycle >= next_render
            action = policy.get_action(observation)
            t0 = time.perf_counter()
            if do_render or not _last_cam_frames:
                # Render cycle: full step with all camera sensors active.
                observation, *_ = task.step(action)
                obs0 = observation[0] if isinstance(observation, list) else observation
                _last_cam_frames = {k: obs0[k] for k in CAMERAS if k in obs0}
            else:
                # Non-render cycle: skip camera rendering for speed.
                # Policy only uses proprioceptive obs (robot_base_pose, tcp_pose_*),
                # so stale camera frames are injected back for API consistency.
                _removed = _pop_camera_sensors(task)
                observation, *_ = task.step(action)
                _restore_camera_sensors(task, _removed)
                obs0 = observation[0] if isinstance(observation, list) else observation
                obs0.update(_last_cam_frames)
            dt_step = time.perf_counter() - t0
            trim_task_caches(task)
            step += 1
            if do_render:
                snap = policy.last_snapshot
                status = {
                    "step": step,
                    "input_fresh": snap.input_fresh,
                    "active_arm": snap.active_arm,
                    "ik_status": policy.ik_status,
                    "policy_latency_ms": policy.last_policy_latency_ms,
                    "gripper_open": snap.gripper_open,
                    "rejected_out_of_order": shared_control.rejected_out_of_order,
                    "rejected_invalid": shared_control.rejected_invalid,
                    "visual_axes": policy.last_axes,
                }
                t0 = time.perf_counter()
                jpeg = compose_frame(observation, args.width, args.height, args.jpeg_quality)
                dt_compose = time.perf_counter() - t0
                shared_frame.publish(jpeg, status)
                frames_sent += 1
                server_fps_elapsed += dt_step + dt_compose
                next_render = time.monotonic() + 1 / args.render_fps  # anchor to now, not cycle start
                if cycle - server_fps_window >= 5.0:
                    avg = server_fps_elapsed / max(frames_sent, 1) * 1000
                    print(f"[fps] sent {frames_sent}f in {cycle-server_fps_window:.1f}s => {frames_sent/(cycle-server_fps_window):.1f} fps | avg {avg:.0f}ms/frame (step={dt_step*1000:.0f}ms+compose={dt_compose*1000:.0f}ms)", flush=True)
                    frames_sent = 0
                    server_fps_elapsed = 0.0
                    server_fps_window = cycle
            if args.duration > 0 and cycle - started >= args.duration:
                break
            stop_event.wait(max(0.0, period - (time.monotonic() - cycle)))
    finally:
        shared_control.disconnect()
        shared_frame.stop()
        if sampler is not None:
            sampler.close()


if __name__ == "__main__":
    main()
