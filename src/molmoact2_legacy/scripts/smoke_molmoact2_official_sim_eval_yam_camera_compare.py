from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from molmo_spaces.data_generation.config.molmoact2_official_sim_eval_yam_box_config import (
    MolmoAct2OfficialSimEvalYamBoxDataGenConfig,
)
from molmo_spaces.data_generation.config.molmoact2_official_yam_box_config import (
    OFFICIAL_YAM_CAMERA_SPECS,
)


def collect_images(summary, prefix, value):
    if isinstance(value, dict):
        for k, v in value.items():
            collect_images(summary, f"{prefix}.{k}" if prefix else str(k), v)
        return
    try:
        arr = np.asarray(value)
    except Exception:
        return
    if arr.dtype == object or arr.ndim < 2 or arr.size <= 1000:
        return
    if arr.ndim >= 3 or "cam" in prefix or "image" in prefix:
        numeric = arr.astype(np.float32, copy=False)
        summary[prefix] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "mean": float(numeric.mean()),
            "std": float(numeric.std()),
            "min": float(numeric.min()),
            "max": float(numeric.max()),
        }


def run_case(label: str, mode: str):
    cfg = MolmoAct2OfficialSimEvalYamBoxDataGenConfig()
    cfg.camera_config = copy.deepcopy(cfg.camera_config)
    if mode == "official_raw":
        for cam in cfg.camera_config.cameras:
            cam.camera_quaternion = list(OFFICIAL_YAM_CAMERA_SPECS[cam.name]["q"])
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    try:
        task = sampler.sample_task(force_advance_scene=True, house_index=0)
        obs = task.get_observations()[0]
        images = {}
        collect_images(images, "", obs)
        cam_poses = {}
        for name, camera in task._env.camera_manager.registry.cameras.items():
            cam_poses[name] = {
                "pos": np.asarray(camera.pos).tolist(),
                "forward": np.asarray(camera.forward).tolist(),
                "up": np.asarray(camera.up).tolist(),
            }
        return {"label": label, "mode": mode, "images": images, "camera_poses": cam_poses}
    finally:
        sampler.close()


def main():
    report = [
        run_case("converted_current", "converted"),
        run_case("official_raw_quat", "official_raw"),
    ]
    out = Path(
        "artifacts/molmospaces/molmoact2_yam_integration/official_sim_eval_bridge_camera_quat_compare.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
