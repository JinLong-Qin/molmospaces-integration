"""Focused regressions for the MolmoAct2 official YAM visual smoke gate."""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

import scripts.smoke_molmoact2_official_sim_eval_yam_request_frames as smoke_gate
from scripts.smoke_molmoact2_official_sim_eval_yam_request_frames import (
    CAMERA_ORDER,
    OFFICIAL_REFERENCE_RELATIVE_DIR,
    collect_static_regression_report,
    validate_request_frame_images,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "examples/molmoact2_official_sim_eval_yam_box/scene.xml"
BASELINE_RUN = (
    REPO_ROOT
    / "artifacts/molmospaces/molmoact2_yam_visual_request_frame_smoke"
    / "20260713T074034.501714Z"
)


def test_active_yam_visual_mjcf_load() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))

    assert model.ngeom >= 18


def test_neutral_lighting_contract() -> None:
    root = ET.parse(SCENE_PATH).getroot()
    headlight = root.find("./visual/headlight")
    assert headlight is not None
    assert [float(value) for value in headlight.get("ambient", "").split()] == pytest.approx(
        [0.3, 0.3, 0.3]
    )
    assert headlight.get("diffuse") == "0 0 0"
    assert headlight.get("specular") == "0 0 0"

    lights = root.findall("./worldbody/light")
    assert [light.get("name") for light in lights] == [
        "front_key_light",
        "overhead_fill_light",
    ]
    for light in lights:
        diffuse = np.asarray([float(value) for value in light.get("diffuse", "").split()])
        assert light.get("directional") == "true"
        assert light.get("castshadow") == "true"
        assert diffuse.shape == (3,)
        assert np.allclose(diffuse, diffuse[0])
        assert 0.0 < diffuse[0] <= 1.0
        assert light.get("ambient") == "0 0 0"
        assert light.get("specular") == "0 0 0"


def test_neutral_background_contract() -> None:
    root = ET.parse(SCENE_PATH).getroot()
    rgba = root.find("./visual/rgba")
    assert rgba is not None
    for attribute in ("haze", "fog"):
        color = np.asarray([float(value) for value in rgba.get(attribute, "").split()])
        assert color.shape == (4,)
        assert np.allclose(color[:3], color[0])
        assert 0.35 <= color[0] <= 0.55
        assert color[3] == 1.0
    assert root.find(".//texture[@type='skybox']") is None
    assert root.find(".//texture[@builtin='gradient']") is None


def test_native_exposure_contract_has_no_image_compensation() -> None:
    smoke_source = (
        REPO_ROOT / "scripts/smoke_molmoact2_official_sim_eval_yam_request_frames.py"
    ).read_text(encoding="utf-8")
    env_source = (REPO_ROOT / "molmo_spaces/env/env.py").read_text(encoding="utf-8")
    forbidden_operations = (
        "adjust_gamma",
        "convertScaleAbs",
        "createTonemap",
        "equalizeHist",
    )
    assert all(operation not in smoke_source for operation in forbidden_operations)
    assert all(operation not in env_source for operation in forbidden_operations)

    root = ET.parse(SCENE_PATH).getroot()
    quality = root.find("./visual/quality")
    visual_map = root.find("./visual/map")
    assert quality is not None
    assert quality.get("shadowsize") == "4096"
    assert quality.get("offsamples") == "4"
    assert visual_map is not None
    assert visual_map.get("znear") == "0.01"
    assert visual_map.get("zfar") == "6"


def test_active_yam_visual_static_model_load() -> None:
    report = collect_static_regression_report(REPO_ROOT)

    assert report["scene"]["model_loaded"] is True


def test_active_yam_visual_static_regression_contract() -> None:
    report = collect_static_regression_report(REPO_ROOT)

    assert report["status"] == "pass"
    assert report["scene"]["headlight_ambient"] == pytest.approx([0.3, 0.3, 0.3])
    assert report["scene"]["active_directional_lights"] == [
        "front_key_light",
        "overhead_fill_light",
    ]
    assert report["scene"]["model_loaded"] is True
    assert report["protected_contract"]["camera_order"] == list(CAMERA_ORDER)
    assert report["protected_contract"]["execution_command_hz"] == 30.0


def test_smoke_module_uses_writable_offline_runtime_caches() -> None:
    assert os.environ["NLTK_DATA"].startswith("/tmp/")
    assert os.environ["MPLCONFIGDIR"].startswith("/tmp/")
    assert os.environ["MOLMOSPACES_ALLOW_WORDNET_FALLBACK"] == "1"
    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["PYOPENGL_PLATFORM"] == "egl"


def test_official_references_are_successful_policy_input_pngs() -> None:
    report = collect_static_regression_report(REPO_ROOT)
    references = report["official_references"]

    assert references["relative_directory"] == OFFICIAL_REFERENCE_RELATIVE_DIR.as_posix()
    assert references["results"]["episode_success"] is True
    assert references["results"]["success_rate"] == 1.0
    assert list(references["images"]) == list(CAMERA_ORDER)
    assert all(item["suffix"] == ".png" for item in references["images"].values())
    assert all("mp4" not in json.dumps(item).lower() for item in references["images"].values())
    assert {
        name: item["sha256"] for name, item in references["images"].items()
    } == smoke_gate.OFFICIAL_REFERENCE_SHA256


def test_official_reference_loader_rejects_mp4_paths(tmp_path: Path) -> None:
    mp4_path = tmp_path / "reference.mp4"
    mp4_path.write_bytes(b"not a policy-input PNG")

    with pytest.raises(ValueError, match="must be PNG"):
        smoke_gate._load_rgb_png(mp4_path)


def test_request_frame_validation_accepts_exact_rgb_camera_order() -> None:
    images = {name: np.zeros((360, 640, 3), dtype=np.uint8) for name in CAMERA_ORDER}

    validated = validate_request_frame_images(images)

    assert list(validated) == list(CAMERA_ORDER)


@pytest.mark.parametrize("missing", CAMERA_ORDER)
def test_request_frame_validation_rejects_missing_camera(missing: str) -> None:
    images = {name: np.zeros((360, 640, 3), dtype=np.uint8) for name in CAMERA_ORDER}
    del images[missing]

    with pytest.raises(ValueError, match=f"Missing request-frame camera: {missing}"):
        validate_request_frame_images(images)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((360, 640), "must be RGB"),
        ((360, 640, 4), "must be RGB"),
        ((640, 360, 3), "must be 640x360"),
    ],
)
def test_request_frame_validation_rejects_malformed_image(
    shape: tuple[int, ...], message: str
) -> None:
    images = {name: np.zeros((360, 640, 3), dtype=np.uint8) for name in CAMERA_ORDER}
    images["left_cam"] = np.zeros(shape, dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        validate_request_frame_images(images)


def test_request_frame_validation_rejects_reordered_cameras() -> None:
    images = {
        "left_cam": np.zeros((360, 640, 3), dtype=np.uint8),
        "top_cam": np.zeros((360, 640, 3), dtype=np.uint8),
        "right_cam": np.zeros((360, 640, 3), dtype=np.uint8),
    }

    with pytest.raises(ValueError, match="Incorrect request-frame camera order"):
        validate_request_frame_images(images)


def test_smoke_writer_only_creates_timestamped_run_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_dir = tmp_path / "official"
    reference_dir.mkdir()
    references = {}
    captured = {}
    for index, name in enumerate(CAMERA_ORDER):
        image = np.full((360, 640, 3), fill_value=40 + index, dtype=np.uint8)
        reference_path = reference_dir / f"ep000_{name}.png"
        smoke_gate.Image.fromarray(image).save(reference_path)
        references[name] = {"path": str(reference_path)}
        captured[name] = image

    monkeypatch.setattr(
        smoke_gate,
        "collect_static_regression_report",
        lambda _root: {
            "status": "pass",
            "official_references": {
                "results": {"episode_success": True, "success_rate": 1.0},
                "images": references,
            },
        },
    )
    monkeypatch.setattr(
        smoke_gate,
        "capture_request_frames",
        lambda _root, seed: (
            captured,
            {
                "policy_service_called": False,
                "policy_inference_called": False,
                "policy_actions_executed": False,
                "action_count": 0,
                "rollout_steps": 0,
                "camera_order": list(CAMERA_ORDER),
                "seed": seed,
            },
        ),
    )

    report_path = smoke_gate.run_smoke_gate(tmp_path, seed=0)

    run_dir = report_path.parent
    assert run_dir.parent == tmp_path / smoke_gate.RUN_ARTIFACT_ROOT
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "comparison_report.json",
        "left_cam.png",
        "right_cam.png",
        "top_cam.png",
    ]
    assert sorted(path.name for path in reference_dir.iterdir()) == [
        "ep000_left_cam.png",
        "ep000_right_cam.png",
        "ep000_top_cam.png",
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["failures"] == []
    assert report["schema_version"] == 1
    assert report["camera_order"] == list(CAMERA_ORDER)
    assert report["ordered_cameras"] == list(CAMERA_ORDER)
    assert report["seed"] == 0
    assert report["no_policy"]["status"] == "pass"
    assert set(report["artifact_files"]) == {
        "top_cam.png",
        "left_cam.png",
        "right_cam.png",
    }


@pytest.mark.parametrize(
    ("runtime", "field"),
    [
        ({}, "policy_service_called"),
        ({"policy_service_called": True}, "policy_service_called"),
        ({"policy_inference_called": True}, "policy_inference_called"),
        ({"policy_actions_executed": True}, "policy_actions_executed"),
        ({"action_count": 1}, "action_count"),
        ({"rollout_steps": 1}, "rollout_steps"),
    ],
)
def test_no_policy_evidence_fails_closed(runtime: dict[str, object], field: str) -> None:
    complete = {
        "policy_service_called": False,
        "policy_inference_called": False,
        "policy_actions_executed": False,
        "action_count": 0,
        "rollout_steps": 0,
    }
    complete.update(runtime)
    if runtime == {}:
        complete.pop(field)

    result = smoke_gate.validate_no_policy_evidence(complete)

    assert result["status"] == "fail"
    assert any(failure["field"] == field for failure in result["failures"])


def test_malformed_capture_still_writes_failure_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed = {name: np.zeros((360, 640, 3), dtype=np.uint8) for name in CAMERA_ORDER}
    malformed["left_cam"] = np.zeros((360, 640, 4), dtype=np.uint8)
    monkeypatch.setattr(smoke_gate, "collect_static_regression_report", lambda _root: {})
    monkeypatch.setattr(
        smoke_gate,
        "capture_request_frames",
        lambda _root, seed: (
            malformed,
            {
                "policy_service_called": False,
                "policy_inference_called": False,
                "policy_actions_executed": False,
                "action_count": 0,
                "rollout_steps": 0,
                "seed": seed,
            },
        ),
    )

    report_path = smoke_gate.run_smoke_gate(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "fail"
    assert any(failure["reason"] == "malformed_request_frame" for failure in report["failures"])
    assert sorted(path.name for path in report_path.parent.iterdir()) == [
        "comparison_report.json",
        "left_cam.png",
        "right_cam.png",
        "top_cam.png",
    ]


def test_cli_exits_nonzero_for_failure_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "comparison_report.json"
    report_path.write_text('{"status": "fail"}\n', encoding="utf-8")
    monkeypatch.setattr(smoke_gate, "run_smoke_gate", lambda seed: report_path)
    monkeypatch.setattr(smoke_gate.sys, "argv", ["smoke-gate"])

    with pytest.raises(SystemExit, match="1"):
        smoke_gate.main()


def test_cli_threads_documented_seed_to_smoke_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "comparison_report.json"
    report_path.write_text('{"status": "pass"}\n', encoding="utf-8")
    observed = {}

    def fake_run_smoke_gate(repo_root: Path | None = None, seed: int = 42) -> Path:
        observed["repo_root"] = repo_root
        observed["seed"] = seed
        return report_path

    monkeypatch.setattr(smoke_gate, "run_smoke_gate", fake_run_smoke_gate)
    monkeypatch.setattr(smoke_gate.sys, "argv", ["smoke-gate", "--seed", "0"])

    smoke_gate.main()

    assert observed == {"repo_root": None, "seed": 0}
    assert capsys.readouterr().out.strip() == str(report_path)


def test_validate_report_accepts_exact_passing_artifact(tmp_path: Path) -> None:
    artifact_files = {}
    for name in CAMERA_ORDER:
        path = tmp_path / f"{name}.png"
        smoke_gate.Image.fromarray(np.zeros((360, 640, 3), dtype=np.uint8)).save(path)
        artifact_files[path.name] = smoke_gate._sha256(path)
    report = {
        "status": "pass",
        "camera_order": list(CAMERA_ORDER),
        "seed": 0,
        "runtime": {
            "policy_service_called": False,
            "policy_inference_called": False,
            "policy_actions_executed": False,
            "action_count": 0,
            "rollout_steps": 0,
            "seed": 0,
        },
        "artifact_files": artifact_files,
        "comparison": {name: {"status": "pass", "predicates": {}} for name in CAMERA_ORDER},
        "official_references": {
            "results": {"episode_success": True, "success_rate": 1.0},
            "images": {
                name: {"path": f"official/ep000_{name}.png", "suffix": ".png"}
                for name in CAMERA_ORDER
            },
        },
        "human_video_used": False,
    }
    report_path = tmp_path / "comparison_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    validated = smoke_gate.validate_report(report_path)

    assert validated["status"] == "pass"


def test_validate_report_rejects_nonpassing_or_extra_artifacts(tmp_path: Path) -> None:
    report_path = tmp_path / "comparison_report.json"
    report_path.write_text('{"status": "fail"}\n', encoding="utf-8")
    (tmp_path / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="status must be pass"):
        smoke_gate.validate_report(report_path)


def test_cli_validate_report_does_not_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "comparison_report.json"
    report_path.write_text('{"status": "pass"}\n', encoding="utf-8")
    monkeypatch.setattr(smoke_gate, "validate_report", lambda path: {"status": "pass"})
    monkeypatch.setattr(
        smoke_gate,
        "run_smoke_gate",
        lambda *args, **kwargs: pytest.fail("validation must not capture"),
    )
    monkeypatch.setattr(
        smoke_gate.sys,
        "argv",
        ["smoke-gate", "--validate-report", str(report_path)],
    )

    smoke_gate.main()

    assert capsys.readouterr().out.strip() == str(report_path)


def _independent_metrics(captured: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    captured_rgb = captured.astype(np.float32)
    reference_rgb = reference.astype(np.float32)
    captured_mean = np.round(captured_rgb.mean(axis=(0, 1)), 6)
    reference_mean = np.round(reference_rgb.mean(axis=(0, 1)), 6)
    luminance_weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    captured_luminance = captured_rgb @ luminance_weights
    reference_luminance = reference_rgb @ luminance_weights
    percentiles = [5, 25, 50, 75, 95]

    cdf_distances = []
    for channel in range(3):
        captured_histogram = np.bincount(captured[..., channel].ravel(), minlength=256).astype(
            np.float64
        )
        reference_histogram = np.bincount(reference[..., channel].ravel(), minlength=256).astype(
            np.float64
        )
        captured_cdf = np.cumsum(captured_histogram / captured[..., channel].size)
        reference_cdf = np.cumsum(reference_histogram / reference[..., channel].size)
        cdf_distances.append(np.abs(captured_cdf[:-1] - reference_cdf[:-1]).mean())

    def texture_coverage(luminance: np.ndarray) -> float:
        roi = luminance[18:270, 64:576]
        horizontal = np.diff(roi, axis=1)[:-1]
        vertical = np.diff(roi, axis=0)[:, :-1]
        magnitude = np.hypot(horizontal, vertical)
        return float(np.mean((magnitude >= 2.0) & (magnitude <= 20.0)))

    return {
        "mean_rgb_l2_delta": float(np.linalg.norm(captured_mean - reference_mean)),
        "luminance_percentile_mae": float(
            np.mean(
                np.abs(
                    np.percentile(captured_luminance, percentiles)
                    - np.percentile(reference_luminance, percentiles)
                )
            )
        ),
        "rgb_cdf_distance": float(np.mean(cdf_distances)),
        "tabletop_texture_coverage_delta": abs(
            texture_coverage(captured_luminance) - texture_coverage(reference_luminance)
        ),
    }


def test_metric_implementation_matches_independent_recomputation() -> None:
    reference = np.arange(360 * 640 * 3, dtype=np.uint32).reshape(360, 640, 3)
    reference = (reference % 256).astype(np.uint8)
    captured = np.roll(reference, shift=17, axis=1)

    observed = smoke_gate.compute_appearance_metrics(captured, reference)
    expected = _independent_metrics(captured, reference)

    assert observed == pytest.approx(expected)


@pytest.mark.parametrize(
    ("metric", "absolute_limit", "baseline", "relative_fraction"),
    [
        ("mean_rgb_l2_delta", 65.0, 80.0, 0.70),
        ("luminance_percentile_mae", 45.0, 60.0, 0.70),
        ("rgb_cdf_distance", 0.18, 0.20, 0.70),
        ("tabletop_texture_coverage_delta", 0.25, 0.30, 0.60),
    ],
)
def test_threshold_equality_passes_and_just_over_fails(
    metric: str, absolute_limit: float, baseline: float, relative_fraction: float
) -> None:
    relative_limit = baseline * relative_fraction
    equality = min(absolute_limit, relative_limit)

    passed = smoke_gate.evaluate_metric_threshold(metric, equality, baseline)
    failed = smoke_gate.evaluate_metric_threshold(metric, equality + 1e-12, baseline)

    assert passed["status"] == "pass"
    assert failed["status"] == "fail"
    assert failed["absolute_limit"] == absolute_limit
    assert failed["baseline_value"] == baseline
    assert failed["relative_limit"] == pytest.approx(relative_limit)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_metric_threshold_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        smoke_gate.evaluate_metric_threshold("mean_rgb_l2_delta", value, 85.0)


@pytest.mark.parametrize(
    ("captured", "reference", "message"),
    [
        (
            np.zeros((360, 640, 3), dtype=np.float32),
            np.zeros((360, 640, 3), dtype=np.uint8),
            "uint8",
        ),
        (np.zeros((360, 640), dtype=np.uint8), np.zeros((360, 640, 3), dtype=np.uint8), "shape"),
        (np.zeros((360, 640, 3), dtype=np.uint8), np.zeros((360, 639, 3), dtype=np.uint8), "shape"),
    ],
)
def test_metric_inputs_reject_malformed_arrays(
    captured: np.ndarray, reference: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        smoke_gate.compute_appearance_metrics(captured, reference)


def test_baseline_hashes_metrics_and_overall_gate_failure() -> None:
    expected_hashes = {
        "top_cam": "974543d6803b9767b7ee11e0f200a4e19c25fecb6391fe56127651d519778f6d",
        "left_cam": "f176ceebea447f17e4486ee2aeecad26e6a1b62fe79a51ed9682fe1dd261a5b2",
        "right_cam": "2735bbfe1f67907f339e9ba159723b05199f9b6387cd38d0dc94a1c756518e58",
    }
    expected_metrics = {
        "top_cam": (85.33326580734972, 45.88228, 0.208656, 0.642861),
        "left_cam": (93.60922198968713, 72.09308, 0.274256, 0.416385),
        "right_cam": (95.31396960255097, 71.23012, 0.267947, 0.438894),
    }
    reference_report = collect_static_regression_report(REPO_ROOT)["official_references"]
    captured = {}
    for name in CAMERA_ORDER:
        path = BASELINE_RUN / f"{name}.png"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hashes[name]
        captured[name] = np.asarray(smoke_gate.Image.open(path)).copy()

    comparison = smoke_gate.build_appearance_comparison(captured, reference_report)
    report_status = smoke_gate.derive_report_status(
        contract_pass=True,
        no_policy_pass=True,
        reference_pass=True,
        comparison=comparison,
    )

    for name, expected in expected_metrics.items():
        metrics = comparison[name]["metrics"]
        assert metrics["mean_rgb_l2_delta"] == expected[0]
        assert metrics["luminance_percentile_mae"] == pytest.approx(expected[1], abs=1e-5)
        assert metrics["rgb_cdf_distance"] == pytest.approx(expected[2], abs=5e-7)
        assert metrics["tabletop_texture_coverage_delta"] == pytest.approx(expected[3], abs=5e-7)
    assert report_status["status"] == "fail"
    assert report_status["failures"]
    assert all(
        {
            "metric",
            "camera",
            "observed_value",
            "absolute_limit",
            "baseline_value",
            "relative_limit",
            "reason",
        }
        <= failure.keys()
        for failure in report_status["failures"]
    )


def test_report_status_requires_contract_no_policy_reference_and_metrics() -> None:
    comparison = {
        name: {
            "predicates": {
                metric: {"status": "pass"} for metric in smoke_gate.APPEARANCE_THRESHOLDS
            }
        }
        for name in CAMERA_ORDER
    }
    assert (
        smoke_gate.derive_report_status(
            contract_pass=True,
            no_policy_pass=True,
            reference_pass=True,
            comparison=comparison,
        )["status"]
        == "pass"
    )

    for field in ("contract_pass", "no_policy_pass", "reference_pass"):
        inputs = dict(contract_pass=True, no_policy_pass=True, reference_pass=True)
        inputs[field] = False
        result = smoke_gate.derive_report_status(comparison=comparison, **inputs)
        assert result["status"] == "fail"
        assert result["failures"][0]["reason"] == field.removesuffix("_pass")
