"""Gate 1D-2B: read-only localhost browser bridge for three camera views.

Diagnostic only. This script doesn't step the task, execute robot actions, or save a demo.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import signal
import struct
import threading
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from check_dual_object_reachability import (
    CAMERAS,
    DualPickupDiagnosticSampler,
    build_config,
)


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MolmoSpaces Gate 1D-2B</title>
<style>
body{margin:0;background:#151719;color:#eee;font:14px system-ui,sans-serif}header{padding:10px 14px;background:#202326;display:flex;gap:18px;align-items:center}b{color:#70d6a8}main{padding:12px}img{display:block;width:100%;max-width:1280px;background:#000;border:1px solid #444}#err{color:#ff8d8d}
</style></head><body><header><strong>Gate 1D-2B read-only</strong><span id="status">connecting</span><span id="fps"></span><span id="age"></span><span id="err"></span></header><main><img id="view" alt="three camera stream"></main>
<script>
const status=document.querySelector('#status'),fps=document.querySelector('#fps'),age=document.querySelector('#age'),err=document.querySelector('#err'),view=document.querySelector('#view');
let lastSeq=-1,lastAt=0,frames=0,windowAt=performance.now();
function connect(){const ws=new WebSocket(`ws://${location.host}/stream`);ws.binaryType='arraybuffer';
ws.onopen=()=>{status.innerHTML='<b>connected</b>';err.textContent=''};
ws.onmessage=e=>{const data=e.data,header=new DataView(data,0,12);lastSeq=header.getUint32(0);lastAt=performance.now();const blob=new Blob([data.slice(12)],{type:'image/jpeg'});const old=view.src;view.src=URL.createObjectURL(blob);if(old)URL.revokeObjectURL(old);frames++;const now=performance.now();if(now-windowAt>=1000){fps.textContent=`${(frames*1000/(now-windowAt)).toFixed(1)} FPS`;frames=0;windowAt=now}};
ws.onclose=()=>{status.textContent='disconnected';setTimeout(connect,1000)};ws.onerror=()=>{err.textContent='socket error'};}
setInterval(()=>{if(lastAt)age.textContent=`frame ${lastSeq}, age ${Math.round(performance.now()-lastAt)} ms`},250);connect();
</script></body></html>"""


@dataclass
class SharedFrame:
    condition: threading.Condition = field(default_factory=threading.Condition)
    sequence: int = 0
    captured_at: float = 0.0
    jpeg: bytes = b""
    stopped: bool = False

    def publish(self, payload: bytes) -> None:
        with self.condition:
            self.sequence += 1
            self.captured_at = time.time()
            self.jpeg = payload
            self.condition.notify_all()

    def snapshot_after(self, sequence: int, timeout: float = 1.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > sequence or self.stopped, timeout=timeout
            )
            return self.sequence, self.captured_at, self.jpeg, self.stopped

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()


def compose_frame(env, width: int, height: int, quality: int) -> bytes:
    env.camera_manager.registry.update_all_cameras(env)
    panels = []
    for camera_name in CAMERAS:
        frame = env.render_rgb_frame(camera_name)
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


async def run_server(shared: SharedFrame, host: str, port: int) -> None:
    async def process_request(connection: ServerConnection, request):
        if request.path == "/healthz":
            return connection.respond(200, "OK\n")
        if request.path == "/":
            response = connection.respond(200, HTML)
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response
        if request.path == "/favicon.ico":
            response = connection.respond(204, "")
            response.headers["Content-Type"] = "image/x-icon"
            return response
        if request.path == "/stream":
            return None
        return connection.respond(404, "Not Found\n")

    async def handler(websocket: ServerConnection) -> None:
        if websocket.request.path != "/stream":
            await websocket.close(code=1008, reason="unknown websocket path")
            return
        sequence = -1
        try:
            while True:
                sequence, captured_at, payload, stopped = await asyncio.to_thread(
                    shared.snapshot_after, sequence
                )
                if stopped:
                    return
                if not payload:
                    continue
                header = struct.pack(">Id", sequence, captured_at)
                await websocket.send(header + payload)
        except ConnectionClosed:
            return

    async with serve(
        handler,
        host,
        port,
        compression=None,
        max_size=None,
        process_request=process_request,
        ping_interval=20,
        ping_timeout=20,
    ) as server:
        print(f"bridge=http://{host}:{port}", flush=True)
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--house-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Gate 1D-2B must bind loopback only")
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    config = build_config(args.seed)
    sampler = DualPickupDiagnosticSampler(config)
    shared = SharedFrame()
    server_thread = threading.Thread(
        target=lambda: asyncio.run(run_server(shared, args.host, args.port)),
        name="gate-1d2b-websocket",
        daemon=True,
    )
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    try:
        task = sampler.sample_task(force_advance_scene=True, house_index=args.house_index)
        if task is None:
            raise RuntimeError("sample_task returned None")
        env = sampler.env
        mujoco.mj_forward(env.current_model, env.current_data)
        server_thread.start()
        started = time.monotonic()
        period = 1.0 / args.fps
        while not stop_event.is_set():
            cycle_started = time.monotonic()
            shared.publish(compose_frame(env, args.width, args.height, args.jpeg_quality))
            if args.duration > 0 and cycle_started - started >= args.duration:
                break
            stop_event.wait(max(0.0, period - (time.monotonic() - cycle_started)))
    finally:
        shared.stop()
        sampler.close()


if __name__ == "__main__":
    main()
