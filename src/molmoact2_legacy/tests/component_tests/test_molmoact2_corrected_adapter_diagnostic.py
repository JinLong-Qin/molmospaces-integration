from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

diagnostic = importlib.import_module(
    "scripts.diagnose_molmoact2_official_sim_eval_yam_corrected_adapter"
)


def test_argument_defaults_are_bounded_and_attach_to_existing_act() -> None:
    args = diagnostic.parse_args([])

    assert args.endpoint == "http://127.0.0.1:8202/act"
    assert args.max_steps == 30
    assert args.action_scale == 4.0
    assert args.right_j3_offset == -0.08
    assert args.artifact_root == "artifacts/molmospaces/molmoact2_yam_integration"
    assert args.gripper_continuity_guard_steps == 0
    assert args.structured_step_diagnostics is False


def test_argument_parser_rejects_rollout_lengths_above_30() -> None:
    with pytest.raises(SystemExit):
        diagnostic.parse_args(["--max-steps", "31"])


def test_corrected_action_scales_arm_deltas_and_sets_right_joint_3_offset() -> None:
    state = np.zeros(14, dtype=np.float32)
    raw = np.zeros(14, dtype=np.float32)
    raw[0] = 0.01
    raw[5] = -0.02
    raw[7] = 0.03
    raw[10] = 0.04
    raw[6] = 0.7
    raw[13] = 0.8

    transformed = diagnostic.apply_corrected_adapter_candidates(
        raw,
        state,
        action_scale=4.0,
        right_j3_offset=-0.08,
    )

    assert transformed[0] == pytest.approx(0.04)
    assert transformed[5] == pytest.approx(-0.08)
    assert transformed[7] == pytest.approx(0.12)
    assert transformed[diagnostic.RIGHT_J3_ACTION_INDEX] == pytest.approx(-0.08)
    assert transformed[6] == pytest.approx(0.7)
    assert transformed[13] == pytest.approx(0.8)


def test_gripper_continuity_guard_is_disabled_by_default() -> None:
    action = np.zeros(14, dtype=np.float32)
    action[6] = 0.9
    action[13] = 0.8

    guarded, metadata = diagnostic.apply_gripper_continuity_guard(
        action,
        chunk_index=0,
        guard_steps=0,
        transport_active=True,
        last_executed_grippers={"left": 0.2, "right": 0.3},
    )

    assert guarded.tolist() == pytest.approx(action.tolist())
    assert metadata["enabled"] is False
    assert metadata["applied"] is False


def test_gripper_continuity_guard_clamps_opening_during_first_n_transport_steps() -> None:
    action = np.zeros(14, dtype=np.float32)
    action[6] = 0.9
    action[13] = 0.1

    guarded, metadata = diagnostic.apply_gripper_continuity_guard(
        action,
        chunk_index=1,
        guard_steps=3,
        transport_active=True,
        last_executed_grippers={"left": 0.2, "right": 0.3},
    )

    assert guarded[6] == pytest.approx(0.2)
    assert guarded[13] == pytest.approx(0.1)
    assert metadata["applied"] is True
    assert metadata["clamped_sides"] == ["left"]


def test_gripper_continuity_guard_does_not_clamp_without_transport_condition() -> None:
    action = np.zeros(14, dtype=np.float32)
    action[6] = 0.9

    guarded, metadata = diagnostic.apply_gripper_continuity_guard(
        action,
        chunk_index=0,
        guard_steps=3,
        transport_active=False,
        last_executed_grippers={"left": 0.2, "right": 0.3},
    )

    assert guarded.tolist() == pytest.approx(action.tolist())
    assert metadata["applied"] is False


def test_structured_step_diagnostic_schema_serializes_unavailable_force_fields() -> None:
    row = diagnostic.build_structured_step_diagnostic(
        step=4,
        chunk_id=2,
        chunk_index=1,
        raw_action=np.arange(14, dtype=np.float32),
        mapped_action=np.arange(14, dtype=np.float32) + 0.5,
        actual_state14=np.arange(14, dtype=np.float32) + 1.0,
        summary={
            "contacts": {"obj": {"right_gripper_touching": True}},
            "any_gripper_contact": True,
            "held": True,
            "object_poses": {"obj": {"position": [1.0, 2.0, 3.0]}},
            "gripper_poses": {"right_gripper": {"position": [0.5, 2.0, 3.0]}},
        },
        previous_summary={
            "object_poses": {"obj": {"position": [0.9, 2.0, 3.0]}},
            "gripper_poses": {"right_gripper": {"position": [0.5, 2.0, 3.0]}},
        },
        guard={"enabled": True, "applied": True},
        force_diagnostics=None,
    )

    serialized = json.loads(json.dumps(row, allow_nan=False))
    assert serialized["chunk"] == {"id": 2, "index": 1}
    assert serialized["gripper"]["raw"] == {"left": 6.0, "right": 13.0}
    assert serialized["gripper"]["mapped"] == {"left": 6.5, "right": 13.5}
    assert serialized["gripper"]["actual_normalized_opening"] == {
        "left": 7.0,
        "right": 14.0,
    }
    assert serialized["actual_normalized_state"] == pytest.approx(
        (np.arange(14, dtype=np.float32) + 1.0).tolist()
    )
    assert serialized["transport"]["touching"] is True
    assert serialized["transport"]["held"] is True
    assert serialized["relative_motion"]["object_displacement_m"] == pytest.approx(0.1)
    assert "arm_command_delta" not in serialized
    assert serialized["target_minus_post_step_normalized_state"]["left"] == pytest.approx(
        [-0.5] * 6
    )
    assert serialized["forces"]["available"] is False
    assert serialized["forces"]["reason"] == "not_requested"


def test_requested_mujoco_forces_use_explicit_unavailable_marker() -> None:
    result = diagnostic.collect_optional_mujoco_forces(object(), requested=True)

    assert result == {
        "available": False,
        "reason": "mujoco_data_unavailable",
        "source": None,
    }


def test_requested_mujoco_forces_prefer_task_env_current_batch_data() -> None:
    class FakeData:
        actuator_force = np.array([1.0, 2.0])
        efc_force = np.array([3.0, 4.0, 5.0])

    fake_env = type("FakeEnv", (), {"mj_datas": [object(), FakeData()]})()
    task = type("FakeTask", (), {"_env": fake_env, "current_batch_index": 1})()

    result = diagnostic.collect_optional_mujoco_forces(task, requested=True)

    assert result == {
        "available": True,
        "source": "task._env.mj_datas[1]",
        "actuator_force": {
            "available": True,
            "scope": "global_actuator_array",
            "values": [1.0, 2.0],
        },
        "efc_force": {
            "available": True,
            "scope": "global_unscoped_constraint_force_array",
            "values": [3.0, 4.0, 5.0],
        },
    }


def test_summarize_info_extracts_held_from_nested_h5_contact_schema() -> None:
    summary = diagnostic.summarize_info(
        {
            "official_yam_behavior": {
                "contacts": {
                    "obj": {
                        "any_gripper_touching": False,
                        "per_gripper": {
                            "left_gripper": {"touching": False, "held": False},
                            "right_gripper": {"touching": True, "held": True},
                        },
                    }
                }
            }
        }
    )

    assert summary["held"] is True


def test_summarize_info_extracts_nested_touching_without_held() -> None:
    summary = diagnostic.summarize_info(
        {
            "official_yam_behavior": {
                "contacts": {
                    "obj": {
                        "any_gripper_touching": False,
                        "per_gripper": {
                            "left_gripper": {"touching": False, "held": False},
                            "right_gripper": {"touching": True, "held": False},
                        },
                    }
                }
            }
        }
    )

    assert summary["any_gripper_contact"] is True
    assert summary["held"] is False


def test_report_writer_creates_timestamped_artifacts_and_selected_actions(tmp_path) -> None:
    out_dir = diagnostic.make_artifact_dir(tmp_path, timestamp="20260703_010203")
    rows = [
        {
            "step": 1,
            "raw_action": [0.0] * 14,
            "transformed_action": [0.0] * 14,
        }
    ]
    report = {
        "evidence_label": diagnostic.DIAGNOSTIC_EVIDENCE_LABEL,
        "act_response_present": True,
        "selected_action_shape": [1, 14],
        "executed_steps": 1,
        "eef_object_min_distance_m": 0.12,
        "contact_flags": {"any_gripper_contact": False},
        "final_n_in_box": 0,
    }

    paths = diagnostic.write_artifacts(out_dir, report, rows)

    assert out_dir.name == "corrected_adapter_diagnostic_20260703_010203"
    assert paths["report"].name == "report.json"
    assert paths["selected_actions"].name == "selected_actions.jsonl"

    saved_report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert saved_report["evidence_label"] == diagnostic.DIAGNOSTIC_EVIDENCE_LABEL
    assert "diagnostic evidence only" in saved_report["evidence_label"].lower()

    selected_lines = paths["selected_actions"].read_text(encoding="utf-8").splitlines()
    assert len(selected_lines) == 1
    assert json.loads(selected_lines[0])["step"] == 1
