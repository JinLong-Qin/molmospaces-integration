from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REQUIRED_DATASETS = ("actions", "states", "dones", "rewards")
REQUIRED_OBS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
REQUIRED_DATAGEN = ("eef_pose", "target_pose", "gripper_action")


def require_finite(name: str, array: np.ndarray) -> None:
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains non-finite values")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a robomimic/MimicGen source HDF5 produced by the PnP converter."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-demos", type=int)
    args = parser.parse_args()
    if not args.input.is_file() or args.input.stat().st_size == 0:
        raise RuntimeError(f"missing or empty HDF5: {args.input}")

    summary = {"input": str(args.input), "demos": [], "total": 0}
    with h5py.File(args.input, "r") as h5:
        if "data" not in h5:
            raise RuntimeError("missing root data group")
        data = h5["data"]
        demos = sorted(k for k in data if k.startswith("demo_"))
        if args.expected_demos is not None and len(demos) != args.expected_demos:
            raise RuntimeError(f"expected {args.expected_demos} demos, found {len(demos)}")
        for demo_name in demos:
            demo = data[demo_name]
            n = int(demo.attrs.get("num_samples", -1))
            if n <= 0:
                raise RuntimeError(f"{demo_name}: invalid num_samples={n}")
            for key in REQUIRED_DATASETS:
                if key not in demo or len(demo[key]) != n:
                    raise RuntimeError(f"{demo_name}: missing/misaligned {key}")
                require_finite(f"{demo_name}/{key}", np.asarray(demo[key]))
            for key in REQUIRED_OBS:
                if key not in demo["obs"] or len(demo["obs"][key]) != n:
                    raise RuntimeError(f"{demo_name}: missing/misaligned obs/{key}")
                require_finite(f"{demo_name}/obs/{key}", np.asarray(demo["obs"][key]))
            for key in REQUIRED_DATAGEN:
                if key not in demo["datagen_info"] or len(demo["datagen_info"][key]) != n:
                    raise RuntimeError(f"{demo_name}: missing/misaligned datagen_info/{key}")
                require_finite(f"{demo_name}/datagen_info/{key}", np.asarray(demo["datagen_info"][key]))
            if not str(demo.attrs.get("seed_kind", "")).startswith("synthetic_"):
                raise RuntimeError(f"{demo_name}: missing synthetic source provenance")
            summary["demos"].append({
                "demo": demo_name,
                "num_samples": n,
                "house_id": int(demo.attrs["house_id"]),
                    "source_h5": str(demo.attrs["source_h5"]),
                    "source_run_root": str(demo.attrs.get("source_run_root", "")),
                    "traj_index": int(demo.attrs["traj_index"]),
            })
            summary["total"] += n
        if int(data.attrs.get("total", -1)) != summary["total"]:
            raise RuntimeError(f"data.total={data.attrs.get('total')} != {summary['total']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
