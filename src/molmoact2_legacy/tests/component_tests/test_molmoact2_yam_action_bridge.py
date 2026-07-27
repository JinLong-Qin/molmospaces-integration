import ast
import importlib
import sys
import types
from pathlib import Path

import numpy as np

abstract_exp_config = types.ModuleType("molmo_spaces.configs.abstract_exp_config")
abstract_exp_config.MlSpacesExpConfig = object
sys.modules[abstract_exp_config.__name__] = abstract_exp_config

base_policy = types.ModuleType("molmo_spaces.policy.base_policy")
base_policy.InferencePolicy = object
sys.modules[base_policy.__name__] = base_policy

yam_policy = importlib.import_module("molmo_spaces.policy.learned_policy.molmoact2_yam_policy")

OFFICIAL_CONFIG_PATH = (
    Path(__file__).parents[2]
    / "molmo_spaces/data_generation/config/molmoact2_official_sim_eval_yam_box_config.py"
)


def _official_bridge_config_values() -> tuple[float, int]:
    module = ast.parse(OFFICIAL_CONFIG_PATH.read_text())
    config_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MolmoAct2OfficialSimEvalYamBoxDataGenConfig"
    )
    seed_assignment = next(
        node
        for node in config_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "seed"
    )
    policy_assignment = next(
        node
        for node in config_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "policy_config"
    )
    policy_call = policy_assignment.value
    assert isinstance(policy_call, ast.Call)
    execution_command_hz = next(
        keyword.value for keyword in policy_call.keywords if keyword.arg == "execution_command_hz"
    )
    return float(ast.literal_eval(execution_command_hz)), int(
        ast.literal_eval(seed_assignment.value)
    )


class _FakeRobot:
    def update_control(self, command) -> None:
        self.command = command

    def compute_control(self) -> None:
        pass


class _FakeEnv:
    n_batch = 1

    def __init__(self) -> None:
        self.robots = [_FakeRobot()]
        self.step_counts = []

    def step(self, sim_steps: int) -> None:
        self.step_counts.append(sim_steps)


class _FakeTask:
    _n_sim_steps_per_ctrl = 4
    _ctrl_dt_ms = 20.0

    def __init__(self) -> None:
        self.env = _FakeEnv()
        self.episode_step_count = 0
        self.last_action = None
        self._cumulative_reward = np.zeros(1)
        self._num_steps_taken = np.ones(1)
        self.action_cache = []

    def is_done(self) -> np.ndarray:
        return np.array([False])

    def num_steps_taken(self) -> int:
        return 1

    def get_and_cache_all_step_information(self):
        return {}, np.zeros(1), np.array([False]), np.array([False]), [{}]


def test_right_arm_absolute_targets_pass_through_unchanged() -> None:
    action = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.25,
            -2.5,
            3.75,
            -4.125,
            5.5,
            -6.625,
            0.0,
        ],
        dtype=np.float32,
    )

    command = yam_policy.molmoact2_yam_action_to_move_group_command(action)

    np.testing.assert_array_equal(command["right_arm"], action[7:13])


def test_official_bridge_uses_official_execution_rate_and_seed() -> None:
    execution_command_hz, seed = _official_bridge_config_values()

    assert execution_command_hz == 30.0
    assert seed == 42


def test_official_bridge_schedules_commands_at_configured_rate() -> None:
    execution_command_hz, _ = _official_bridge_config_values()
    task = _FakeTask()

    yam_policy.execute_molmoact2_yam_action(
        task,
        {"right_arm": np.zeros(6, dtype=np.float32)},
        execution_mode="sim_eval_step",
        command_hz=execution_command_hz,
    )

    assert task.env.step_counts == [7]
