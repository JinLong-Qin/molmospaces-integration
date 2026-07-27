"""No-server smoke test for MolmoAct2 YAM action execution semantics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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

from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
    MolmoAct2YamPolicy,
    execute_molmoact2_yam_action,
)


class FakeMoveGroup:
    def __init__(self, joint_pos: np.ndarray) -> None:
        self.joint_pos = np.asarray(joint_pos, dtype=np.float32)

    @property
    def noop_ctrl(self) -> np.ndarray:
        return self.joint_pos.copy()


class FakeRobotView:
    def __init__(self) -> None:
        self.move_groups = {
            "left_arm": FakeMoveGroup(np.zeros(6, dtype=np.float32)),
            "right_arm": FakeMoveGroup(np.zeros(6, dtype=np.float32)),
            "left_gripper": FakeMoveGroup(np.zeros(1, dtype=np.float32)),
            "right_gripper": FakeMoveGroup(np.zeros(1, dtype=np.float32)),
        }

    def get_move_group(self, name: str) -> FakeMoveGroup:
        return self.move_groups[name]


class FakeRobot:
    def __init__(self) -> None:
        self.robot_view = FakeRobotView()
        self.commands: list[dict[str, np.ndarray]] = []
        self.compute_calls = 0

    def update_control(self, action: dict[str, np.ndarray]) -> None:
        command = {key: np.asarray(value, dtype=np.float32).copy() for key, value in action.items()}
        self.commands.append(command)
        for key, value in command.items():
            self.robot_view.get_move_group(key).joint_pos = value.copy()

    def compute_control(self) -> None:
        self.compute_calls += 1


class FakeEnv:
    def __init__(self) -> None:
        self._robot = FakeRobot()
        self.robots = [self._robot]
        self.n_batch = 1
        self.step_calls: list[int] = []

    def step(self, n_steps: int = 1) -> None:
        self.step_calls.append(n_steps)


class FakeTask:
    def __init__(self) -> None:
        self.env = FakeEnv()
        self._n_sim_steps_per_ctrl = 3
        self.episode_step_count = 0
        self.last_action = None
        self.action_cache = []
        self._cumulative_reward = np.array([0.0])
        self._num_steps_taken = np.array([0])

    def num_steps_taken(self) -> int:
        return self.episode_step_count

    def is_done(self) -> np.ndarray:
        return np.array([False])

    def get_observations(self) -> list[dict]:
        return [{"status": "observed"}]

    def get_and_cache_all_step_information(self):
        return (
            [{"status": "observed"}],
            np.array([0.0]),
            np.array([False]),
            np.array([False]),
            [{"ok": True}],
        )

    def reset(self):
        return [{"status": "reset"}], [{"reset": True}]

    def step(self, _action):
        raise AssertionError("MolmoAct2-YAM runner path must not call generic task.step()")

    def judge_success(self) -> bool:
        return False


class FakeMolmoAct2Policy:
    uses_molmoact2_yam_execution = True
    execution_joint_step = 0.01
    execution_max_smoothing_steps = 100

    def __init__(self) -> None:
        self.calls = 0
        self.task = None

    def get_action(self, _observation):
        self.calls += 1
        if self.calls > 2:
            return None
        return {
            "left_arm": np.full(6, 0.01 * self.calls, dtype=np.float32),
            "right_arm": np.full(6, -0.01 * self.calls, dtype=np.float32),
            "left_gripper": np.array([0.02], dtype=np.float32),
            "right_gripper": np.array([0.03], dtype=np.float32),
        }


def test_dynamic_smoothing_uses_absolute_targets_without_policy_hold() -> None:
    task = FakeTask()
    target = {
        "left_arm": np.full(6, 0.035, dtype=np.float32),
        "right_arm": np.full(6, -0.02, dtype=np.float32),
        "left_gripper": np.array([0.01], dtype=np.float32),
        "right_gripper": np.array([0.03], dtype=np.float32),
    }

    observation, reward, terminal, truncated, info = execute_molmoact2_yam_action(
        task,
        target,
        joint_step=0.01,
        max_smoothing_steps=100,
    )

    robot = task.env.robots[0]
    assert task.episode_step_count == 1
    assert len(robot.commands) == 3
    assert task.last_action is target
    assert task.action_cache == [target]
    assert task.env.step_calls == [3, 3, 3]
    np.testing.assert_allclose(robot.commands[0]["left_arm"], np.full(6, 0.0, dtype=np.float32))
    np.testing.assert_allclose(robot.commands[-1]["left_arm"], target["left_arm"])
    np.testing.assert_allclose(robot.commands[-1]["right_gripper"], target["right_gripper"])
    assert observation == [{"status": "observed"}]
    np.testing.assert_allclose(reward, np.array([0.0]))
    np.testing.assert_array_equal(terminal, np.array([False]))
    np.testing.assert_array_equal(truncated, np.array([False]))
    assert info == [{"ok": True}]


def test_policy_chunk_actions_feed_execution_sequentially() -> None:
    task = FakeTask()
    first_14d = np.zeros(14, dtype=np.float32)
    second_14d = np.zeros(14, dtype=np.float32)
    first_14d[:6] = 0.01
    first_14d[6] = 0.25
    first_14d[7:13] = 0.02
    first_14d[13] = 0.75
    second_14d[:6] = 0.04
    second_14d[6] = 1.0
    second_14d[7:13] = -0.03
    second_14d[13] = 0.0

    first_action = MolmoAct2YamPolicy.raw_action_to_move_group_command(first_14d)
    second_action = MolmoAct2YamPolicy.raw_action_to_move_group_command(second_14d)

    execute_molmoact2_yam_action(task, first_action, joint_step=0.01, max_smoothing_steps=100)
    execute_molmoact2_yam_action(task, second_action, joint_step=0.01, max_smoothing_steps=100)

    robot = task.env.robots[0]
    assert task.episode_step_count == 2
    np.testing.assert_allclose(task.action_cache[0]["left_gripper"], np.array([0.25 * 0.041]))
    np.testing.assert_allclose(task.action_cache[1]["left_gripper"], np.array([0.041]))
    np.testing.assert_allclose(robot.commands[-1]["left_arm"], np.full(6, 0.04, dtype=np.float32))
    np.testing.assert_allclose(robot.commands[-1]["right_arm"], np.full(6, -0.03, dtype=np.float32))
    np.testing.assert_allclose(
        robot.commands[-1]["right_gripper"], np.array([0.0], dtype=np.float32)
    )


def test_run_pipeline_uses_molmoact2_execution_path() -> None:
    task = FakeTask()
    policy = FakeMolmoAct2Policy()

    success = run_pipeline.MyRolloutRunner.run_single_rollout(
        episode_seed=0,
        task=task,
        policy=policy,
    )

    assert success is False
    assert policy.calls == 3
    assert task.episode_step_count == 2
    assert len(task.action_cache) == 2
    np.testing.assert_allclose(task.action_cache[0]["left_arm"], np.full(6, 0.01))
    np.testing.assert_allclose(task.action_cache[1]["left_arm"], np.full(6, 0.02))


def main() -> None:
    test_dynamic_smoothing_uses_absolute_targets_without_policy_hold()
    test_policy_chunk_actions_feed_execution_sequentially()
    test_run_pipeline_uses_molmoact2_execution_path()
    print("molmoact2_yam_execution_semantics smoke passed")


if __name__ == "__main__":
    main()
