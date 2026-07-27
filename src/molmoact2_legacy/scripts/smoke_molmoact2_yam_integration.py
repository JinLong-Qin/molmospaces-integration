"""Integration wiring smoke for MolmoAct2 YAM in MolmoSpaces.

This is a no-server smoke test. It validates that the CLI/config route selects
the MolmoAct2 YAM policy and that a fake ``/act`` response flows through the
adapter into MolmoSpaces-style YAM actions. It is not a closed-loop benchmark.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/molmospaces-matplotlib")
os.environ.setdefault("NLTK_DATA", "/tmp/molmospaces-nltk-data")

try:
    import nltk

    nltk.download = lambda *args, **kwargs: True
except ImportError:
    pass

import numpy as np

SCRIPTS_DATAGEN = Path(__file__).resolve().parent / "datagen"
sys.path.insert(0, str(SCRIPTS_DATAGEN))

import run_pipeline  # noqa: E402

from molmo_spaces.configs.policy_configs_baselines import (  # noqa: E402
    MolmoAct2YamPolicyConfig,
)
from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (  # noqa: E402
    MolmoAct2YamPolicy,
)


class FakeTask:
    def get_task_description(self) -> str:
        return "Pick up the credit card."


class FakeMolmoAct2Client:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def act(self, request: dict) -> dict:
        self.requests.append(request)
        actions = np.zeros((2, 14), dtype=np.float32)
        actions[0, :6] = np.arange(6, dtype=np.float32) + 0.1
        actions[0, 6] = 0.5
        actions[0, 7:13] = np.arange(6, dtype=np.float32) + 10.1
        actions[0, 13] = 1.0
        actions[1, :6] = np.arange(6, dtype=np.float32) + 20.1
        actions[1, 6] = 0.0
        actions[1, 7:13] = np.arange(6, dtype=np.float32) + 30.1
        actions[1, 13] = 0.25
        return {"actions": actions, "source": "fake_molmoact2_act"}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    run_pipeline.add_run_pipeline_args(parser)
    return parser


def make_observation() -> dict:
    return {
        "exo_camera": np.full((3, 4, 3), 11, dtype=np.uint8),
        "left_wrist_camera": np.full((3, 4, 3), 22, dtype=np.uint8),
        "right_wrist_camera": np.full((3, 4, 3), 33, dtype=np.uint8),
        "qpos": {
            "left_arm": np.arange(6, dtype=np.float32),
            "left_gripper": np.array([0.0205], dtype=np.float32),
            "right_arm": np.arange(10, 16, dtype=np.float32),
            "right_gripper": np.array([0.041], dtype=np.float32),
        },
    }


def test_cli_config_and_fake_act_wiring() -> None:
    args = make_parser().parse_args(["--robot", "bimanual_yam", "--policy", "molmoact2_yam"])
    assert args.robot == "bimanual_yam"
    assert args.policy == "molmoact2_yam"
    assert run_pipeline.resolve_task_type(args) == "packing"

    policy_config = run_pipeline.get_policy_config(args.policy, robot=args.robot)
    assert isinstance(policy_config, MolmoAct2YamPolicyConfig)
    assert policy_config.remote_config["path"] == "/act"
    assert policy_config.instruction_override == "Put everything into the box."
    assert policy_config.camera_mapping == {
        "top_cam": "exo_camera",
        "left_cam": "left_wrist_camera",
        "right_cam": "right_wrist_camera",
    }

    fake_client = FakeMolmoAct2Client()
    exp_config = SimpleNamespace(
        task_type="packing",
        policy_config=policy_config,
        camera_config=SimpleNamespace(cameras=[]),
    )
    policy = policy_config.policy_factory(exp_config, FakeTask(), client=fake_client)
    assert isinstance(policy, MolmoAct2YamPolicy)

    observation = make_observation()
    first_action = policy.get_action(observation)
    second_action = policy.get_action(observation)

    assert len(fake_client.requests) == 1
    request = fake_client.requests[0]
    assert sorted(request.keys()) == [
        "instruction",
        "left_cam",
        "num_steps",
        "right_cam",
        "state",
        "top_cam",
    ]
    assert request["instruction"] == "Put everything into the box."
    assert request["top_cam"].shape == (3, 4, 3)
    assert request["left_cam"].shape == (3, 4, 3)
    assert request["right_cam"].shape == (3, 4, 3)
    assert request["num_steps"] == policy_config.num_steps
    np.testing.assert_allclose(
        request["state"],
        np.array([0, 1, 2, 3, 4, 5, 0.5, 10, 11, 12, 13, 14, 15, 1.0], dtype=np.float32),
    )

    np.testing.assert_allclose(first_action["left_arm"], np.arange(6, dtype=np.float32) + 0.1)
    np.testing.assert_allclose(first_action["right_arm"], np.arange(6, dtype=np.float32) + 10.1)
    np.testing.assert_allclose(first_action["left_gripper"], np.array([0.0205], dtype=np.float32))
    np.testing.assert_allclose(first_action["right_gripper"], np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(second_action["left_arm"], np.arange(6, dtype=np.float32) + 20.1)
    np.testing.assert_allclose(
        second_action["right_gripper"], np.array([0.75 * 0.041], dtype=np.float32)
    )


def test_cli_config_rejects_non_yam_robot() -> None:
    try:
        run_pipeline.get_policy_config("molmoact2_yam", robot="droid")
    except ValueError as exc:
        message = str(exc)
        assert "--policy molmoact2_yam" in message
        assert "--robot bimanual_yam" in message
    else:
        raise AssertionError("molmoact2_yam must reject non-bimanual_yam robots")


def main() -> None:
    test_cli_config_and_fake_act_wiring()
    test_cli_config_rejects_non_yam_robot()
    print("molmoact2_yam_integration wiring smoke passed")
    print("scope: fake-client integration wiring smoke, not closed-loop benchmark")


if __name__ == "__main__":
    main()
