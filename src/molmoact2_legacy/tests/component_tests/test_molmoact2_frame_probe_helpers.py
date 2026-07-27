from __future__ import annotations

import importlib

import numpy as np
import pytest

probe = importlib.import_module("scripts.diagnose_molmoact2_official_sim_eval_yam_frame_probe")


def test_vector_alignment_reports_projection_and_cosine() -> None:
    metrics = probe.vector_alignment_metrics([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])

    assert metrics["displacement_norm"] == pytest.approx(1.0)
    assert metrics["object_direction_norm"] == pytest.approx(2.0)
    assert metrics["projection_m"] == pytest.approx(1.0)
    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["aligned_toward_object"] is True


def test_vector_alignment_reports_antialigned_motion() -> None:
    metrics = probe.vector_alignment_metrics([-0.5, 0.0, 0.0], [2.0, 0.0, 0.0])

    assert metrics["projection_m"] == pytest.approx(-0.5)
    assert metrics["cosine"] == pytest.approx(-1.0)
    assert metrics["aligned_toward_object"] is False


def test_vector_alignment_uses_nulls_for_unmeasurable_zero_vectors() -> None:
    metrics = probe.vector_alignment_metrics([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])

    assert metrics["displacement_norm"] == pytest.approx(0.0)
    assert metrics["object_direction_norm"] == pytest.approx(2.0)
    assert metrics["projection_m"] is None
    assert metrics["cosine"] is None
    assert metrics["aligned_toward_object"] is None


def test_build_arm_probe_step_measures_eef_motion_toward_nearest_object() -> None:
    previous = {
        "gripper_poses": {
            "right_gripper": {"position": [0.0, 0.0, 0.0]},
            "left_gripper": {"position": [0.0, 1.0, 0.0]},
        },
        "object_poses": {
            "obj_a": {"position": [1.0, 0.0, 0.0]},
            "obj_b": {"position": [0.0, 3.0, 0.0]},
        },
    }
    current = {
        "gripper_poses": {
            "right_gripper": {"position": [0.25, 0.0, 0.0]},
            "left_gripper": {"position": [0.0, 0.5, 0.0]},
        },
        "object_poses": {
            "obj_a": {"position": [1.0, 0.0, 0.0]},
            "obj_b": {"position": [0.0, 3.0, 0.0]},
        },
        "gripper_object_distances": {
            "right_gripper": {"obj_a": 0.75, "obj_b": 3.01},
            "left_gripper": {"obj_a": 1.12, "obj_b": 2.5},
        },
        "min_distance_m": 0.75,
    }

    row = probe.build_arm_probe_step(previous, current)

    assert row["distance_delta_m"] is None
    assert row["arms"]["right_gripper"]["selected_object"] == "obj_a"
    assert row["arms"]["right_gripper"]["displacement_norm"] == pytest.approx(0.25)
    assert row["arms"]["right_gripper"]["cosine"] == pytest.approx(1.0)
    assert row["arms"]["left_gripper"]["selected_object"] == "obj_b"
    assert row["arms"]["left_gripper"]["projection_m"] == pytest.approx(-0.5)


def test_build_arm_probe_step_records_distance_delta_when_available() -> None:
    previous = {
        "gripper_poses": {"right_gripper": {"position": [0.0, 0.0, 0.0]}},
        "object_poses": {"obj_a": {"position": [1.0, 0.0, 0.0]}},
        "gripper_object_distances": {"right_gripper": {"obj_a": 1.0}},
        "min_distance_m": 1.0,
    }
    current = {
        "gripper_poses": {"right_gripper": {"position": [0.2, 0.0, 0.0]}},
        "object_poses": {"obj_a": {"position": [1.0, 0.0, 0.0]}},
        "gripper_object_distances": {"right_gripper": {"obj_a": 0.8}},
        "min_distance_m": 0.8,
    }

    row = probe.build_arm_probe_step(previous, current)

    assert row["distance_delta_m"] == pytest.approx(-0.2)
    assert row["distance_trend"] == "decreased"


def test_aggregate_alignment_metrics_counts_nulls_and_distance_trends() -> None:
    rows = [
        {
            "distance_delta_m": -0.2,
            "distance_trend": "decreased",
            "arms": {
                "right_gripper": {
                    "displacement_norm": 0.25,
                    "projection_m": 0.25,
                    "cosine": 1.0,
                    "aligned_toward_object": True,
                }
            },
        },
        {
            "distance_delta_m": 0.1,
            "distance_trend": "increased",
            "arms": {
                "right_gripper": {
                    "displacement_norm": 0.1,
                    "projection_m": -0.1,
                    "cosine": -1.0,
                    "aligned_toward_object": False,
                },
                "left_gripper": {
                    "displacement_norm": None,
                    "projection_m": None,
                    "cosine": None,
                    "aligned_toward_object": None,
                },
            },
        },
    ]

    aggregates = probe.aggregate_alignment_metrics(rows)

    right = aggregates["right_gripper"]
    assert right["measured_steps"] == 2
    assert right["aligned_steps"] == 1
    assert right["mean_cosine"] == pytest.approx(0.0)
    assert right["mean_projection_m"] == pytest.approx(0.075)
    assert right["mean_displacement_norm_m"] == pytest.approx(0.175)
    assert aggregates["left_gripper"]["missing_steps"] == 1
    assert aggregates["distance_trends"] == {
        "decreased": 1,
        "increased": 1,
        "unchanged": 0,
        "unknown": 0,
    }


def test_build_arm_probe_step_prefers_intended_object_when_available() -> None:
    previous = {
        "gripper_poses": {"right_gripper": {"position": [0.0, 0.0, 0.0]}},
        "object_poses": {
            "near_obj": {"position": [0.1, 0.0, 0.0]},
            "intended_obj": {"position": [0.0, 1.0, 0.0]},
        },
        "min_distance_m": 0.1,
    }
    current = {
        "gripper_poses": {"right_gripper": {"position": [0.0, 0.2, 0.0]}},
        "object_poses": {
            "near_obj": {"position": [0.1, 0.0, 0.0]},
            "intended_obj": {"position": [0.0, 1.0, 0.0]},
        },
        "gripper_object_distances": {
            "right_gripper": {"near_obj": 0.1, "intended_obj": 0.8},
        },
        "min_distance_m": 0.1,
    }

    row = probe.build_arm_probe_step(
        previous,
        current,
        {"right_gripper": "intended_obj"},
    )

    arm = row["arms"]["right_gripper"]
    assert arm["selected_object"] == "intended_obj"
    assert arm["selection_method"] == "intended_object"
    assert arm["selected_distance_m"] == pytest.approx(0.8)
    assert arm["cosine"] == pytest.approx(1.0)


def test_apply_named_adapter_option_negates_right_local_delta_dim_2() -> None:
    base = [0.0] * 14
    raw = [0.0] * 14
    raw[7] = 0.01
    raw[8] = 0.02
    raw[9] = 0.03
    raw[10] = 0.04

    transformed = probe.apply_named_adapter_option(
        raw,
        base,
        action_scale=4.0,
        right_j3_offset=-0.08,
        adapter_option="right_j2_delta_negated",
    )

    assert transformed[7] == pytest.approx(0.04)
    assert transformed[8] == pytest.approx(0.08)
    assert transformed[9] == pytest.approx(-0.12)
    assert transformed[10] == pytest.approx(-0.08)


def test_apply_named_adapter_option_bilateral_j2_delta_negated_negates_both_arms() -> None:
    base = [0.0] * 14
    raw = [0.0] * 14
    # left arm local dims 0..2 -> action indices 0..2
    raw[0] = 0.01
    raw[1] = 0.02
    raw[2] = 0.03
    # right arm local dims 0..2 -> action indices 7..9
    raw[7] = 0.01
    raw[8] = 0.02
    raw[9] = 0.03
    raw[10] = 0.04

    transformed = probe.apply_named_adapter_option(
        raw,
        base,
        action_scale=4.0,
        right_j3_offset=-0.08,
        adapter_option="bilateral_j2_delta_negated",
    )

    # Left arm: dims 0 and 1 keep sign, dim 2 negated
    assert transformed[0] == pytest.approx(0.04)
    assert transformed[1] == pytest.approx(0.08)
    assert transformed[2] == pytest.approx(-0.12)
    # Right arm behavior matches right_j2_delta_negated for arm dims 0..2
    assert transformed[7] == pytest.approx(0.04)
    assert transformed[8] == pytest.approx(0.08)
    assert transformed[9] == pytest.approx(-0.12)
    # right_j3 offset still applied
    assert transformed[10] == pytest.approx(-0.08)


def test_apply_named_adapter_option_rejects_unknown_option() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError):
        probe.apply_named_adapter_option(
            [0.0] * 14,
            [0.0] * 14,
            action_scale=4.0,
            right_j3_offset=-0.08,
            adapter_option="nope",
        )


def test_append_video_frames_accepts_observation_sequence() -> None:
    frames = {}
    observation = [
        {
            "top_cam": [[[1, 2, 3]]],
            "left_cam": [[[4, 5, 6]]],
            "right_cam": [[[7, 8, 9]]],
        }
    ]

    probe._append_video_frames(observation, frames)

    assert sorted(frames) == ["left_cam", "right_cam", "top_cam"]
    assert frames["top_cam"][0].shape == (1, 1, 3)


class _SyntheticChunkPolicy:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = iter(chunks)
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.inference_call_count = 0

    def inference_model(self, _request: dict) -> np.ndarray:
        if self.actions_buffer is None or self.current_buffer_index >= len(self.actions_buffer):
            self.actions_buffer = list(next(self._chunks))
            self.current_buffer_index = 0
            self.inference_call_count += 1
        action = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return action


def _two_chunks() -> list[np.ndarray]:
    first = np.zeros((2, 14), dtype=np.float32)
    first[:, 6] = [0.2, 0.2]
    first[:, 13] = [0.3, 0.3]
    second = np.zeros((4, 14), dtype=np.float32)
    second[:, 6] = [0.9, 0.8, 0.7, 0.6]
    second[:, 13] = [0.9, 0.8, 0.7, 0.6]
    return [first, second]


def test_multi_chunk_guard_default_off_preserves_actions_and_indices() -> None:
    policy = _SyntheticChunkPolicy(_two_chunks())
    chunk_id = -1
    observed = []
    for _ in range(6):
        raw, chunk_id, chunk_index, _new_chunk = probe.infer_buffered_action_with_chunk_metadata(
            policy, {}, chunk_id=chunk_id
        )
        guarded, _metadata = probe.apply_gripper_continuity_guard(
            raw,
            chunk_index=chunk_index,
            guard_steps=0,
            transport_active=True,
            last_executed_grippers={"left": 0.2, "right": 0.3},
        )
        observed.append((chunk_id, chunk_index, guarded[6], guarded[13]))

    assert observed == pytest.approx(
        [
            (0, 0, 0.2, 0.3),
            (0, 1, 0.2, 0.3),
            (1, 0, 0.9, 0.9),
            (1, 1, 0.8, 0.8),
            (1, 2, 0.7, 0.7),
            (1, 3, 0.6, 0.6),
        ]
    )


def test_multi_chunk_guard_clamps_only_new_chunk_prefix_during_transport() -> None:
    policy = _SyntheticChunkPolicy(_two_chunks())
    chunk_id = -1
    observed = []
    last_grippers = {"left": 0.2, "right": 0.3}
    for _ in range(6):
        raw, chunk_id, chunk_index, new_chunk = probe.infer_buffered_action_with_chunk_metadata(
            policy, {}, chunk_id=chunk_id
        )
        guarded, metadata = probe.apply_gripper_continuity_guard(
            raw,
            chunk_index=chunk_index,
            guard_steps=2 if chunk_id > 0 else 0,
            transport_active=True,
            last_executed_grippers=last_grippers,
        )
        observed.append(
            (chunk_id, chunk_index, new_chunk, guarded[6], guarded[13], metadata["applied"])
        )
        last_grippers = {"left": float(guarded[6]), "right": float(guarded[13])}

    assert observed == pytest.approx(
        [
            (0, 0, True, 0.2, 0.3, False),
            (0, 1, False, 0.2, 0.3, False),
            (1, 0, True, 0.2, 0.3, True),
            (1, 1, False, 0.2, 0.3, True),
            (1, 2, False, 0.7, 0.7, False),
            (1, 3, False, 0.6, 0.6, False),
        ]
    )
