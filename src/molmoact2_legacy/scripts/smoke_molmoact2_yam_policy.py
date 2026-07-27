"""No-server smoke test for MolmoAct2 YAM policy conversion."""

from __future__ import annotations

import base64
import os
import shutil
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

from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
    MOLMOACT2_YAM_STATE_ACTION_ORDER,
    MolmoAct2YamPolicy,
    build_molmoact2_yam_request,
    parse_molmoact2_actions,
    scale_molmoact2_yam_gripper_action,
)


class FakeTask:
    def get_task_description(self) -> str:
        return "Pick up the credit card."


class FakeClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.calls = 0

    def act(self, request: dict) -> dict:
        self.requests.append(request)
        self.calls += 1
        actions = np.stack(
            [
                np.linspace(0.0, 1.3, 14, dtype=np.float32),
                np.linspace(1.0, 2.3, 14, dtype=np.float32),
            ]
        )
        actions[0, 6] = 0.25
        actions[0, 13] = 0.75
        actions[1, 6] = 1.0
        actions[1, 13] = 0.0
        return {"actions": actions, "dt_ms": 1.25}

    def reset(self) -> None:
        self.requests.clear()
        self.calls = 0


def make_obs() -> dict:
    return {
        "exo_camera": np.full((4, 5, 3), 10, dtype=np.uint8),
        "left_wrist_camera": np.full((4, 5, 3), 20, dtype=np.uint8),
        "right_wrist_camera": np.full((4, 5, 3), 30, dtype=np.uint8),
        "qpos": {
            "left_arm": np.arange(6, dtype=np.float32),
            "left_gripper": np.array([0.0205], dtype=np.float32),
            "right_arm": np.arange(10, 16, dtype=np.float32),
            "right_gripper": np.array([0.041], dtype=np.float32),
        },
    }


def make_policy(fake_client: FakeClient) -> MolmoAct2YamPolicy:
    debug_dump_dir = Path("/tmp/molmospaces-molmoact2-yam-smoke-debug")
    if debug_dump_dir.exists():
        shutil.rmtree(debug_dump_dir)

    exp_config = SimpleNamespace(
        task_type="packing",
        policy_config=SimpleNamespace(
            force_enable_depth=False,
            remote_config=None,
            endpoint_url="http://unused/act",
            request_timeout=0.1,
            num_steps=7,
            n_action_steps=2,
            gripper_max=0.041,
            instruction_override="Put everything into the box.",
            debug_dump_dir=str(debug_dump_dir),
            debug_dump_max_calls=2,
            camera_mapping={
                "top_cam": "exo_camera",
                "left_cam": "left_wrist_camera",
                "right_cam": "right_wrist_camera",
            },
        ),
        camera_config=SimpleNamespace(cameras=[]),
    )
    policy = MolmoAct2YamPolicy(exp_config, task=FakeTask(), client=fake_client)
    return policy


def test_pure_request_and_response_helpers() -> None:
    obs = make_obs()
    request = build_molmoact2_yam_request(
        obs,
        instruction="Put everything in the box.",
        num_steps=7,
        gripper_max=0.041,
        camera_mapping={
            "top_cam": "exo_camera",
            "left_cam": "left_wrist_camera",
            "right_cam": "right_wrist_camera",
        },
    )
    assert sorted(request.keys()) == [
        "instruction",
        "left_cam",
        "num_steps",
        "right_cam",
        "state",
        "top_cam",
    ]
    assert request["state"].shape == (14,)
    assert MOLMOACT2_YAM_STATE_ACTION_ORDER == (
        "left_arm_6",
        "left_gripper",
        "right_arm_6",
        "right_gripper",
    )
    np.testing.assert_allclose(
        request["state"],
        np.array([0, 1, 2, 3, 4, 5, 0.5, 10, 11, 12, 13, 14, 15, 1.0], dtype=np.float32),
    )

    expected_actions = np.arange(28, dtype=np.float32).reshape(2, 14)
    actions = parse_molmoact2_actions({"actions": expected_actions})
    assert actions.shape == (2, 14)

    json_numpy_actions = {
        "__numpy__": base64.b64encode(expected_actions.tobytes()).decode("ascii"),
        "dtype": expected_actions.dtype.str,
        "shape": list(expected_actions.shape),
    }
    np.testing.assert_allclose(
        parse_molmoact2_actions({"actions": json_numpy_actions}),
        expected_actions,
    )

    single = parse_molmoact2_actions({"actions": np.arange(14, dtype=np.float32)})
    assert single.shape == (1, 14)


def test_policy_buffers_fake_client_actions() -> None:
    fake_client = FakeClient()
    policy = make_policy(fake_client)
    obs = make_obs()

    first = policy.get_action(obs)
    second = policy.get_action(obs)

    assert fake_client.calls == 1
    sent = fake_client.requests[0]
    assert sent["num_steps"] == 7
    assert sent["instruction"] == "Put everything into the box."
    assert sent["top_cam"].shape == (4, 5, 3)
    assert sent["state"].shape == (14,)

    debug_dump_dir = Path(policy.debug_dump_dir)
    dumped = sorted(path.name for path in debug_dump_dir.iterdir())
    assert dumped == [
        "act_000000_left_cam.ppm",
        "act_000000_metadata.json",
        "act_000000_right_cam.ppm",
        "act_000000_top_cam.ppm",
    ]

    np.testing.assert_allclose(first["left_arm"], np.linspace(0.0, 1.3, 14, dtype=np.float32)[:6])
    np.testing.assert_allclose(first["left_gripper"], np.array([0.25 * 0.041], dtype=np.float32))
    np.testing.assert_allclose(first["right_gripper"], np.array([0.75 * 0.041], dtype=np.float32))
    np.testing.assert_allclose(second["left_gripper"], np.array([0.041], dtype=np.float32))
    np.testing.assert_allclose(second["right_gripper"], np.array([0.0], dtype=np.float32))


def test_molmoact2_high_gripper_scalar_opens_yam_gripper() -> None:
    np.testing.assert_allclose(
        scale_molmoact2_yam_gripper_action(0.99, gripper_max=0.041),
        np.array([0.99 * 0.041], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        scale_molmoact2_yam_gripper_action(-1.0, gripper_max=0.041),
        np.array([0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        scale_molmoact2_yam_gripper_action(2.0, gripper_max=0.041),
        np.array([0.041], dtype=np.float32),
    )


def main() -> None:
    test_pure_request_and_response_helpers()
    test_policy_buffers_fake_client_actions()
    test_molmoact2_high_gripper_scalar_opens_yam_gripper()
    print("molmoact2_yam_policy smoke passed")


if __name__ == "__main__":
    main()
