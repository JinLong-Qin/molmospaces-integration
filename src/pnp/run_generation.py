#!/usr/bin/env python3
"""Parameterized MimicGen/MolmoSpaces rollout runner.

Experiment-specific launchers should provide defaults and call this CLI. The
runner owns target iteration, source selection, artifact inspection, deduplication,
JSONL provenance, and summary generation.
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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_keys(path: Path, expected: int | None) -> list[str]:
    import h5py

    with h5py.File(path, "r") as handle:
        keys = sorted(handle["data"], key=lambda value: int(value.rsplit("_", 1)[1]))
    if expected is not None and len(keys) != expected:
        raise ValueError(f"expected {expected} source demos, found {len(keys)}")
    return keys


def target_range(path: Path, start: int, end: int) -> range:
    count = len(json.loads(path.read_text())["seeds"])
    end = count - 1 if end < 0 else end
    if start < 0 or end < start or end >= count:
        raise ValueError(f"invalid target range: {start}-{end}; manifest has {count}")
    return range(start, end + 1)


def inspect_attempt(
    work: Path,
    name: str,
    target: int,
    mode: str,
    source_key: str | None,
    code: int,
    source_sha: str,
    manifest_sha: str,
    layout_sha: str,
    seen_actions: set[str],
    seen_layouts: set[str],
) -> dict:
    artifact = work / "artifacts/mimicgen_pnp" / name
    result_file = artifact / "generate_result.json"
    if not result_file.exists():
        result_file = artifact / "result.json"
    trace_file = artifact / "success_trace.json"
    action_file = artifact / "generated_actions.npy"
    replay_hdf5 = artifact / "generated_replay_package.hdf5"
    videos = sorted(artifact.glob("*.mp4"))
    row = {
        "name": name,
        "mode": mode,
        "requested_target": target,
        "requested_source_demo_key": source_key,
        "exit_code": code,
        "artifact_dir": str(artifact),
        "source_hdf5_sha256": source_sha,
        "target_manifest_sha256": manifest_sha,
        "layout_sha256": layout_sha,
        "has_result": result_file.exists(),
        "continuity_validated": False,
        "generated_hdf5_persisted": False,
        "eligible_for_training_hdf5": False,
    }
    if not result_file.exists():
        return row
    result = json.loads(result_file.read_text())
    trace = json.loads(trace_file.read_text()) if trace_file.exists() else []
    selected = [int(value) for value in result.get("src_demo_inds", [])]
    whole_source = mode == "whole-source" and not result.get("select_src_per_subtask") and selected and len(set(selected)) == 1
    per_subtask = mode == "per-subtask" and bool(result.get("select_src_per_subtask"))
    tail30 = len(trace) >= 30 and all(bool(value.get("success")) for value in trace[-30:])
    action_sha = digest(action_file) if action_file.exists() else None
    videos_ok = len(videos) >= 2 and all(video.stat().st_size > 0 for video in videos)
    target_matches = result.get("generation_env_seed_index") == target
    direct_hdf5 = Path(result["direct_hdf5"]).resolve() if result.get("direct_hdf5") else None
    generated_hdf5_persisted = bool(
        replay_hdf5.is_file()
        and replay_hdf5.stat().st_size > 0
        and direct_hdf5 is not None
        and direct_hdf5.is_file()
        and direct_hdf5.stat().st_size > 0
    )
    strict = bool(code == 0 and (whole_source or per_subtask) and result.get("final_success") and result.get("success_persistent_to_end") and result.get("post_hold_steps") == 30 and tail30 and videos_ok and action_sha and target_matches and generated_hdf5_persisted)
    duplicate_action = bool(strict and action_sha in seen_actions)
    duplicate_layout = bool(strict and layout_sha in seen_layouts)
    row.update({
        "target_seed_index": result.get("generation_env_seed_index"),
        "house_id": result.get("generation_env_house_id"),
        "src_demo_inds": selected,
        "distinct_src_demo_count": len(set(selected)),
        "whole_source_verified": bool(whole_source),
        "selection_mode_verified": bool(per_subtask),
        "final_success": bool(result.get("final_success")),
        "success_persistent_to_end": bool(result.get("success_persistent_to_end")),
        "tail30_success": tail30,
        "videos_ok": videos_ok,
        "video_count": len(videos),
        "replay_hdf5": str(replay_hdf5),
        "direct_hdf5": str(direct_hdf5) if direct_hdf5 else None,
        "generated_hdf5_persisted": generated_hdf5_persisted,
        "eligible_for_training_hdf5": generated_hdf5_persisted,
        "num_actions": result.get("num_actions_executed"),
        "action_sha256": action_sha,
        "target_matches": target_matches,
        "strict_success": strict,
        "duplicate_action": duplicate_action,
        "duplicate_layout": duplicate_layout,
        "accepted_unique": bool(strict and not duplicate_action and not duplicate_layout),
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("MOLMOSPACES_ROOT", "."))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--work", required=True)
    parser.add_argument("--source-hdf5", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--mode", choices=("per-subtask", "whole-source"), required=True)
    parser.add_argument("--source-count", type=int)
    parser.add_argument("--target-success", type=int, default=100)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--target-end", type=int, default=-1)
    parser.add_argument("--rng-seed-base", type=int, default=10000)
    parser.add_argument("--run-label", default="mimicgen_generation")
    parser.add_argument("--run-dir")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--extra-rollout-arg", action="append", default=[])
    args = parser.parse_args()

    root = Path(args.root).resolve()
    work = Path(args.work).resolve()
    source = Path(args.source_hdf5).resolve()
    manifest_path = Path(args.target_manifest).resolve()
    if not source.is_file() or not source.stat().st_size:
        raise SystemExit(f"missing source HDF5: {source}")
    if not manifest_path.is_file() or not manifest_path.stat().st_size:
        raise SystemExit(f"missing target manifest: {manifest_path}")
    source_sha = digest(source)
    manifest_sha = digest(manifest_path)
    keys = source_keys(source, args.source_count)
    targets = target_range(manifest_path, args.target_start, args.target_end)
    manifest = json.loads(manifest_path.read_text())
    run_dir = Path(args.run_dir).resolve() if args.run_dir else work / "logs" / f"{args.run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts_file = run_dir / "attempts.jsonl"
    accepted_file = run_dir / "accepted.jsonl"
    action_hashes = run_dir / "action_hashes.txt"
    layout_hashes = run_dir / "layout_hashes.txt"
    log_file = run_dir / "collector.log"
    for path in (attempts_file, accepted_file, action_hashes, layout_hashes):
        path.touch(exist_ok=True)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    (work / "logs" / f"latest_{args.run_label}_logdir.txt").write_text(str(run_dir) + "\n")

    def log(message: str) -> None:
        line = f"{datetime.now().astimezone().isoformat()} {message}"
        print(line, flush=True)
        with log_file.open("a") as output:
            output.write(line + "\n")

    seen_actions = set(action_hashes.read_text().split())
    seen_layouts = set(layout_hashes.read_text().split())
    max_attempts = args.max_attempts or (len(targets) if args.mode == "whole-source" else max(len(targets), args.target_success * 3))
    log(f"START mode={args.mode} targets={targets.start}-{targets.stop - 1} source_demos={len(keys)} diagnostic={args.diagnostic}")
    generated_hdf5 = work / "artifacts/mimicgen_pnp" / f"{args.run_label}_generated.hdf5"
    for attempt in range(max_attempts):
        accepted = read_jsonl(accepted_file)
        if len(accepted) >= args.target_success:
            break
        target = targets.start + attempt % len(targets)
        rng_seed = args.rng_seed_base + attempt
        source_key = keys[target % len(keys)] if args.mode == "whole-source" else None
        name = f"{args.run_label}_target{target:03d}_rng{rng_seed:05d}"
        if any(row.get("name") == name for row in read_jsonl(attempts_file)):
            continue
        command = [args.python, str(root / "src/pnp/generate_pick_place_rollout.py"), "--seed-index", str(target), "--out-name", name, "--source-hdf5", str(source), "--target-manifest", str(manifest_path), "--demo-keys", source_key or ",".join(keys), "--mimicgen-rng-seed", str(rng_seed), "--transform-first-robot-pose", "--post-hold-steps", "30", "--save-videos", "--direct-hdf5", str(generated_hdf5), *args.extra_rollout_arg]
        if args.mode == "per-subtask":
            command.append("--select-src-per-subtask")
        log(f"START name={name} target={target} source={source_key or 'pool'}")
        with (run_dir / f"{name}.log").open("w") as output:
            completed = subprocess.run(
                command,
                cwd=root,
                env={**os.environ, "MOLMOSPACES_PNP_WORKDIR": str(work)},
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        row = inspect_attempt(work, name, target, args.mode, source_key, completed.returncode, source_sha, manifest_sha, manifest["seeds"][target]["layout_sha256"], seen_actions, seen_layouts)
        with attempts_file.open("a") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row.get("accepted_unique"):
            row["accepted_index"] = len(read_jsonl(accepted_file)) + 1
            with accepted_file.open("a") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen_actions.add(row["action_sha256"])
            seen_layouts.add(row["layout_sha256"])
            action_hashes.write_text("\n".join(sorted(seen_actions)) + "\n")
            layout_hashes.write_text("\n".join(sorted(seen_layouts)) + "\n")
            log(f"ACCEPT name={name} accepted={row['accepted_index']}")
        else:
            log(f"REJECT name={name} code={completed.returncode}")
    attempts = read_jsonl(attempts_file)
    accepted = read_jsonl(accepted_file)
    summary = {
        "mode": args.mode, "diagnostic": args.diagnostic, "run_dir": str(run_dir),
        "source_hdf5": str(source), "source_hdf5_sha256": source_sha,
        "target_manifest": str(manifest_path), "target_manifest_sha256": manifest_sha,
        "source_demo_count": len(keys), "target_range": [targets.start, targets.stop - 1],
        "target_success": args.target_success, "max_attempts": max_attempts,
        "attempts": len(attempts), "strict_successes": sum(bool(row.get("strict_success")) for row in attempts),
        "accepted_unique": len(accepted),
        "generated_hdf5": str(generated_hdf5),
        "generated_hdf5_persisted": generated_hdf5.is_file() and generated_hdf5.stat().st_size > 0,
        "eligible_for_training_hdf5": len(accepted) >= args.target_success and generated_hdf5.is_file() and generated_hdf5.stat().st_size > 0,
        "complete": len(accepted) >= args.target_success,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    log("FINISH " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
