#!/usr/bin/env python3
"""Run one bounded public-shard source-to-rollout smoke."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def call(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        code = subprocess.run(
            command, stdout=output, stderr=subprocess.STDOUT, env=env, check=False
        ).returncode
    if code:
        raise SystemExit(f"command failed ({code}); see {log}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-shard", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=20)
    args = parser.parse_args()
    root = args.root.resolve()
    work = args.work.resolve()
    py = str(Path(args.python).resolve())
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MOLMOSPACES_ROOT"] = str(root)
    env["MOLMOSPACES_PNP_WORKDIR"] = str(work)
    env["PYTHONPATH"] = ":".join(
        filter(
            None,
            [
                str(root),
                str(root / "vendor/mimicgen"),
                str(root / "vendor/robomimic"),
                env.get("PYTHONPATH", ""),
            ],
        )
    )

    manifest = work / "source_manifest.json"
    replay_root = work / "replay"
    source_hdf5 = work / "source.hdf5"
    call(
        [
            py,
            str(root / "src/pnp/run_source_hdf5_pipeline.py"),
            "--root",
            str(root),
            "--python",
            py,
            "--source-shard",
            str(args.source_shard.resolve()),
            "--work",
            str(work),
            "--candidate-count",
            "1",
            "--manifest",
            str(manifest),
            "--replay-root",
            str(replay_root),
            "--output-hdf5",
            str(source_hdf5),
            "--run-dir",
            str(work / "source_pipeline"),
        ],
        work / "source_pipeline.log",
        env,
    )
    row = json.loads(manifest.read_text())["seeds"][0]
    raw_dir = Path(row["raw_h5_dir"])
    raw_hdf5 = (
        (work / "artifacts/seeds" / raw_dir) if not raw_dir.is_absolute() else raw_dir
    ) / row["raw_h5"]
    import h5py
    import numpy as np

    with h5py.File(raw_hdf5, "r") as source:
        group = source[row["traj_key"]]
        scene = json.loads(bytes(group["obs_scene"][()]).rstrip(b"\0"))
        robot_base_pose = np.asarray(group["obs/extra/robot_base_pose"][0], dtype=float).tolist()
        place_receptacle_uid = str(scene["place_receptacle_name"]).rsplit("/", 1)[-1]
    target_manifest = work / "target_manifest.json"
    call(
        [
            py,
            str(root / "src/pnp/sample_fixedbase_target_manifest.py"),
            "--output",
            str(target_manifest),
            "--count",
            str(args.target_count),
            "--max-attempts",
            str(args.max_attempts),
            "--house-id",
            str(row["house_id"]),
            "--pickup-object",
            str(scene["object_name"]),
            "--place-receptacle-uid",
            place_receptacle_uid,
            "--robot-base-pose",
            *map(str, robot_base_pose),
            "--source-manifest",
            str(manifest),
        ],
        work / "target_sample.log",
        env,
    )
    call(
        [
            py,
            str(root / "src/pnp/validate_fixedbase_target_manifest.py"),
            "--input",
            str(target_manifest),
            "--expected-count",
            str(args.target_count),
            "--house-id",
            str(row["house_id"]),
            "--robot-base-pose",
            *map(str, robot_base_pose),
        ],
        work / "target_validate.log",
        env,
    )
    config = work / "generation.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mimicgen_generation",
                "run": {
                    "repository_root": str(root),
                    "python": py,
                    "work_dir": str(work / "generation"),
                    "label": "public_shard_smoke",
                },
                "inputs": {
                    "source_hdf5": str(source_hdf5),
                    "target_manifest": str(target_manifest),
                    "source_count": 1,
                },
                "generation": {
                    "mode": "per-subtask",
                    "target_success": 1,
                    "max_attempts": 1,
                    "target_start": 0,
                    "target_end": 0,
                    "rng_seed_base": 10000,
                    "diagnostic": False,
                    "extra_rollout_args": [],
                },
            },
            indent=2,
        )
        + "\n"
    )
    call(
        [py, str(root / "src/pnp/run_experiment.py"), "--config", str(config)],
        work / "generation.log",
        env,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "work": str(work),
                "source_hdf5": str(source_hdf5),
                "target_manifest": str(target_manifest),
                "generation_config": str(config),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
