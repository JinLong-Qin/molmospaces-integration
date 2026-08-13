#!/usr/bin/env python3
"""Validate and execute a configuration-driven Pick-and-Place experiment.

The configuration is JSON to avoid an additional parser dependency. This module
contains experiment orchestration only; simulator/data semantics remain in the
specialized Python CLIs it invokes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is incomplete or ambiguous."""


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError("configuration root must be an object")
    return value


def require_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be an object")
    return value


def require_string(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty string")
    return value


def resolve(config_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def build_generation_command(config_path: Path, config: dict[str, Any]) -> list[str]:
    if config.get("schema_version") != 1:
        raise ConfigurationError("schema_version must be 1")
    if config.get("kind") != "mimicgen_generation":
        raise ConfigurationError("kind must be mimicgen_generation")

    run = require_object(config, "run")
    inputs = require_object(config, "inputs")
    generation = require_object(config, "generation")
    root = resolve(config_path, require_string(run, "repository_root"))
    runner = root / "src/pnp/run_generation.py"
    if not runner.is_file():
        raise ConfigurationError(
            f"repository_root does not contain {runner.relative_to(root)}: {root}"
        )

    mode = generation.get("mode")
    if mode not in {"per-subtask", "whole-source"}:
        raise ConfigurationError("generation.mode must be per-subtask or whole-source")
    if mode == "whole-source" and generation.get("diagnostic") is not True:
        raise ConfigurationError("whole-source is diagnostic-only; set generation.diagnostic=true")
    source_count = inputs.get("source_count")
    if source_count is not None and (not isinstance(source_count, int) or source_count < 1):
        raise ConfigurationError("inputs.source_count must be a positive integer when provided")

    python = str(run.get("python") or sys.executable)
    command = [
        python,
        str(runner),
        "--root",
        str(root),
        "--python",
        python,
        "--work",
        str(resolve(config_path, require_string(run, "work_dir"))),
        "--source-hdf5",
        str(resolve(config_path, require_string(inputs, "source_hdf5"))),
        "--target-manifest",
        str(resolve(config_path, require_string(inputs, "target_manifest"))),
        "--mode",
        mode,
        "--target-success",
        str(generation.get("target_success", 1)),
        "--target-start",
        str(generation.get("target_start", 0)),
        "--target-end",
        str(generation.get("target_end", -1)),
        "--rng-seed-base",
        str(generation.get("rng_seed_base", 10000)),
        "--run-label",
        str(run.get("label", "pnp_generation")),
    ]
    if source_count is not None:
        command.extend(["--source-count", str(source_count)])
    if generation.get("max_attempts") is not None:
        command.extend(["--max-attempts", str(generation["max_attempts"])])
    if generation.get("diagnostic", False):
        command.append("--diagnostic")
    extra = generation.get("extra_rollout_args", [])
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ConfigurationError("generation.extra_rollout_args must be a list of strings")
    for item in extra:
        command.append(f"--extra-rollout-arg={item}")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to a JSON experiment configuration."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the command without running it.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    try:
        command = build_generation_command(config_path, load_config(config_path))
    except ConfigurationError as exc:
        parser.error(str(exc))
    print(json.dumps({"config": str(config_path), "command": command}, indent=2))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
