#!/usr/bin/env python3
"""Parameterized source replay/conversion pipeline for MolmoSpaces PnP.

The domain operations remain in the existing selector, replay collector,
converter, and validator. This CLI owns orchestration and provenance only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(command: list[str], log: Path, env: dict[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False, env=env)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datagen-root", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--house-id", required=True)
    parser.add_argument("--run-name-prefix", required=True)
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--allow-nonpersistent-candidates", action="store_true")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--output-hdf5", required=True)
    parser.add_argument("--action-type", default="joint_position")
    parser.add_argument("--run-dir")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    work = Path(args.work).resolve()
    manifest = Path(args.manifest).resolve()
    replay_root = Path(args.replay_root).resolve()
    output_hdf5 = Path(args.output_hdf5).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else work / "logs" / f"source_hdf5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    replay_root.mkdir(parents=True, exist_ok=True)
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    py = str(Path(args.python).resolve())
    selector = root / "src/pnp/select_source_pool.py"
    replay = root / "src/pnp/replay_source_candidate.py"
    converter = root / "src/pnp/convert_source_hdf5.py"
    validator = root / "src/pnp/validate_robomimic_source_hdf5.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(filter(None, [str(root), str(root / "vendor/mimicgen"), str(root / "vendor/robomimic"), env.get("PYTHONPATH", "")]))

    command = [py, str(selector), "--franka-datagen-root", args.datagen_root, "--house-id", str(args.house_id), "--run-name-prefix", args.run_name_prefix, "--count", str(args.candidate_count), "--out", str(manifest)]
    if args.allow_nonpersistent_candidates:
        command.append("--allow-nonpersistent-candidates")
    code = run(command, run_dir / "select.log", env)
    if code:
        raise SystemExit(f"source selection failed with exit code {code}; see {run_dir / 'select.log'}")

    failed = 0
    for index in range(args.candidate_count):
        log = run_dir / f"replay_seed_{index:02d}.log"
        code = run([py, str(replay), "--seed-index", str(index), "--manifest", str(manifest), "--out-root", str(replay_root)], log, env)
        failed += code != 0

    rows = []
    accepted: list[int] = []
    for index in range(args.candidate_count):
        result_path = replay_root / f"seed_{index:02d}" / "datagen_info_collection_result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {"error": "missing replay result"}
        hard_pass = bool(result.get("final_success") and result.get("success_persistent_to_end"))
        accepted += [index] if hard_pass else []
        rows.append({"seed_index": index, "hard_pass": hard_pass, "result_path": str(result_path), "result": result})
    acceptance = {"candidate_count": len(rows), "replay_failed_process_count": failed, "hard_pass_count": len(accepted), "accepted_indices": accepted, "rows": rows}
    (run_dir / "replay_acceptance.json").write_text(json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n")
    if not accepted:
        raise SystemExit(f"no replay hard-pass source trajectories; see {run_dir / 'replay_acceptance.json'}")

    accepted_csv = ",".join(map(str, accepted))
    code = run([py, str(converter), "--accepted", accepted_csv, "--action-type", args.action_type, "--manifest", str(manifest), "--replay-root", str(replay_root), "--out", str(output_hdf5)], run_dir / "convert.log", env)
    if code:
        raise SystemExit(f"source conversion failed with exit code {code}; see {run_dir / 'convert.log'}")
    code = run([py, str(validator), "--input", str(output_hdf5), "--expected-demos", str(len(accepted))], run_dir / "validate.log", env)
    if code:
        raise SystemExit(f"source validation failed with exit code {code}; see {run_dir / 'validate.log'}")
    artifacts = {"manifest": str(manifest), "replay_root": str(replay_root), "output_hdf5": str(output_hdf5), "accepted_indices": accepted, "hdf5_sha256": hashlib.sha256(output_hdf5.read_bytes()).hexdigest(), "action_type": args.action_type}
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(artifacts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
