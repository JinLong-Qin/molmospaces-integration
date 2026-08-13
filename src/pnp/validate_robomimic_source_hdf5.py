from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REQUIRED_DATASETS = ("actions", "states", "dones", "rewards")
REQUIRED_OBS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
REQUIRED_DATAGEN = ("eef_pose", "target_pose", "gripper_action")
ACTION_AUDIT_DATASETS = {
    "joint_position": ("source_joint_position_actions", 8),
    "tcp_delta": ("source_tcp_delta_actions", 7),
}


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
        action_type = str(h5.attrs.get("action_type", "joint_position"))
        if action_type not in ACTION_AUDIT_DATASETS:
            raise RuntimeError(f"unsupported action_type={action_type!r}")
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
            for key, width in ACTION_AUDIT_DATASETS.values():
                if key not in demo or demo[key].shape != (n, width):
                    raise RuntimeError(
                        f"{demo_name}: expected {key} shape {(n, width)}, "
                        f"found {demo[key].shape if key in demo else None}"
                    )
                require_finite(f"{demo_name}/{key}", np.asarray(demo[key]))
            selected_key, selected_width = ACTION_AUDIT_DATASETS[action_type]
            if demo["actions"].shape != (n, selected_width):
                raise RuntimeError(
                    f"{demo_name}: action_type={action_type} requires actions shape "
                    f"{(n, selected_width)}, found {demo['actions'].shape}"
                )
            if not np.array_equal(demo["actions"][:], demo[selected_key][:]):
                raise RuntimeError(
                    f"{demo_name}: actions do not equal selected audit dataset {selected_key}"
                )
            if str(demo.attrs.get("action_type", "")) != action_type:
                raise RuntimeError(f"{demo_name}: action_type attribute mismatch")
            for key in REQUIRED_OBS:
                if key not in demo["obs"] or len(demo["obs"][key]) != n:
                    raise RuntimeError(f"{demo_name}: missing/misaligned obs/{key}")
                require_finite(f"{demo_name}/obs/{key}", np.asarray(demo["obs"][key]))
            for key in REQUIRED_DATAGEN:
                if key not in demo["datagen_info"] or len(demo["datagen_info"][key]) != n:
                    raise RuntimeError(f"{demo_name}: missing/misaligned datagen_info/{key}")
                require_finite(
                    f"{demo_name}/datagen_info/{key}", np.asarray(demo["datagen_info"][key])
                )
            if not str(demo.attrs.get("seed_kind", "")).startswith("synthetic_"):
                raise RuntimeError(f"{demo_name}: missing synthetic source provenance")
            summary["demos"].append(
                {
                    "demo": demo_name,
                    "num_samples": n,
                    "house_id": int(demo.attrs["house_id"]),
                    "source_h5": str(demo.attrs["source_h5"]),
                    "source_run_root": str(demo.attrs.get("source_run_root", "")),
                    "traj_index": int(demo.attrs["traj_index"]),
                }
            )
            summary["total"] += n
        if int(data.attrs.get("total", -1)) != summary["total"]:
            raise RuntimeError(f"data.total={data.attrs.get('total')} != {summary['total']}")
        summary["action_type"] = action_type
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
