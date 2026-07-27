"""No-policy request-frame smoke gate for the active official YAM scene."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("NLTK_DATA", "/tmp/molmospaces-nltk-data")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/molmospaces-matplotlib")
os.environ.setdefault("MOLMOSPACES_ALLOW_WORDNET_FALLBACK", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from PIL import Image

CAMERA_ORDER = ("top_cam", "left_cam", "right_cam")
IMAGE_SIZE = (640, 360)
SCENE_RELATIVE_PATH = Path("examples/molmoact2_official_sim_eval_yam_box/scene.xml")
CONFIG_RELATIVE_PATH = Path(
    "molmo_spaces/data_generation/config/molmoact2_official_sim_eval_yam_box_config.py"
)
BASE_CONFIG_RELATIVE_PATH = Path(
    "molmo_spaces/data_generation/config/molmoact2_official_yam_box_config.py"
)
OFFICIAL_REFERENCE_RELATIVE_DIR = Path(
    "sim_eval/outputs/official_yam_bf16_chunk_seed_search_20260704_190153/seed_0/20260704_190157"
)
OFFICIAL_REFERENCE_FRAME_DIR = (
    OFFICIAL_REFERENCE_RELATIVE_DIR / "frames/BimanualYAMPutEverythingInBox-v1"
)
OFFICIAL_REPO_ARTIFACT_SUFFIX = Path(
    "artifacts/molmospaces/molmoact2_yam_integration_analysis_20260620_142443/molmoact2"
)
RUN_ARTIFACT_ROOT = Path("artifacts/molmospaces/molmoact2_yam_visual_request_frame_smoke")
REPORT_SCHEMA_VERSION = 1
SCENE_SEED = 0
REJECTED_BASELINE_RELATIVE_DIR = RUN_ARTIFACT_ROOT / "20260713T074034.501714Z"
LUMINANCE_PERCENTILES = (5, 25, 50, 75, 95)
TABLETOP_ROI = {"x": (0.10, 0.90), "y": (0.05, 0.75)}
OFFICIAL_REFERENCE_SHA256 = {
    "top_cam": "9ec1a41ecd21e693bef0d604bc66fbfc39fc035d592e21d1bab3b28d3134f2e7",
    "left_cam": "b5b4b492a4ccfd53968967a7b0796cea0c1ddbba1e17a0b8b6fc710eb32b34cf",
    "right_cam": "7adfd73378ba9fdec000cdc1f42d899c7ee0f2750aa19df1a8d530fbbd4ab3cc",
}
REJECTED_BASELINE_SHA256 = {
    "top_cam": "974543d6803b9767b7ee11e0f200a4e19c25fecb6391fe56127651d519778f6d",
    "left_cam": "f176ceebea447f17e4486ee2aeecad26e6a1b62fe79a51ed9682fe1dd261a5b2",
    "right_cam": "2735bbfe1f67907f339e9ba159723b05199f9b6387cd38d0dc94a1c756518e58",
}
REJECTED_BASELINE_METRICS = {
    "top_cam": {
        "mean_rgb_l2_delta": 85.33326580734972,
        "luminance_percentile_mae": 45.88228,
        "rgb_cdf_distance": 0.208656,
        "tabletop_texture_coverage_delta": 0.642861,
    },
    "left_cam": {
        "mean_rgb_l2_delta": 93.60922198968713,
        "luminance_percentile_mae": 72.09308,
        "rgb_cdf_distance": 0.274256,
        "tabletop_texture_coverage_delta": 0.416385,
    },
    "right_cam": {
        "mean_rgb_l2_delta": 95.31396960255097,
        "luminance_percentile_mae": 71.23012,
        "rgb_cdf_distance": 0.267947,
        "tabletop_texture_coverage_delta": 0.438894,
    },
}
APPEARANCE_THRESHOLDS = {
    "mean_rgb_l2_delta": {"absolute_limit": 65.0, "baseline_fraction": 0.70},
    "luminance_percentile_mae": {"absolute_limit": 45.0, "baseline_fraction": 0.70},
    "rgb_cdf_distance": {"absolute_limit": 0.18, "baseline_fraction": 0.70},
    "tabletop_texture_coverage_delta": {
        "absolute_limit": 0.25,
        "baseline_fraction": 0.60,
    },
}
EXPECTED_CAMERA_SPECS = {
    "top_cam": {
        "mount": "bimanual_base",
        "resolution": (640, 360),
        "hfov_deg": 69.4,
        "p": [0.15, 0.0, 0.8],
        "q": [0.7660444431189782, 0.0, 0.6427876096865391, 0.0],
    },
    "left_cam": {
        "mount": "left_link_6",
        "resolution": (640, 360),
        "hfov_deg": 87.0,
        "p": [0.0, 0.09, 0.06],
        "q": [
            0.612372429196013,
            -0.35355339154618404,
            -0.3535533966987049,
            -0.612372438120441,
        ],
    },
    "right_cam": {
        "mount": "right_link_6",
        "resolution": (640, 360),
        "hfov_deg": 87.0,
        "p": [0.0, 0.09, 0.06],
        "q": [
            0.612372429196013,
            -0.35355339154618404,
            -0.3535533966987049,
            -0.612372438120441,
        ],
    },
}

EXPECTED_PHYSICAL_GEOMS = {
    "floor": {
        "type": "plane",
        "size": "2 2 0.1",
        "pos": "0 0 0",
    },
    "obj_073-a_lego_duplo_base": {
        "type": "box",
        "size": "0.040 0.030 0.025",
        "mass": "0.035",
        "friction": "1.0 0.01 0.001",
    },
    "obj_073-a_lego_duplo_stud_1": {
        "type": "cylinder",
        "size": "0.010 0.006",
        "pos": "-0.018 -0.010 0.031",
        "mass": "0.001",
        "friction": "1.0 0.01 0.001",
    },
    "obj_073-a_lego_duplo_stud_2": {
        "type": "cylinder",
        "size": "0.010 0.006",
        "pos": "0.018 -0.010 0.031",
        "mass": "0.001",
        "friction": "1.0 0.01 0.001",
    },
    "obj_073-a_lego_duplo_stud_3": {
        "type": "cylinder",
        "size": "0.010 0.006",
        "pos": "-0.018 0.010 0.031",
        "mass": "0.001",
        "friction": "1.0 0.01 0.001",
    },
    "obj_073-a_lego_duplo_stud_4": {
        "type": "cylinder",
        "size": "0.010 0.006",
        "pos": "0.018 0.010 0.031",
        "mass": "0.001",
        "friction": "1.0 0.01 0.001",
    },
    "obj_056_tennis_ball_geom": {
        "type": "sphere",
        "size": "0.033",
        "mass": "0.058",
        "friction": "1.0 0.01 0.001",
    },
    "open_box_floor": {
        "type": "box",
        "size": "0.098 0.098 0.004",
        "pos": "0 0 0.004",
        "contype": "1",
        "conaffinity": "1",
    },
    "open_box_wall_pos_x": {
        "type": "box",
        "size": "0.004 0.098 0.030",
        "pos": "0.094 0 0.038",
        "contype": "1",
        "conaffinity": "1",
    },
    "open_box_wall_neg_x": {
        "type": "box",
        "size": "0.004 0.098 0.030",
        "pos": "-0.094 0 0.038",
        "contype": "1",
        "conaffinity": "1",
    },
    "open_box_wall_pos_y": {
        "type": "box",
        "size": "0.090 0.004 0.030",
        "pos": "0 0.094 0.038",
        "contype": "1",
        "conaffinity": "1",
    },
    "open_box_wall_neg_y": {
        "type": "box",
        "size": "0.090 0.004 0.030",
        "pos": "0 -0.094 0.038",
        "contype": "1",
        "conaffinity": "1",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_float_vector(value: str | None, *, name: str) -> list[float]:
    if value is None:
        raise AssertionError(f"Missing {name}")
    return [float(item) for item in value.split()]


def _find_config_class(module: ast.Module) -> ast.ClassDef:
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MolmoAct2OfficialSimEvalYamBoxDataGenConfig"
    )


def _class_assignment(config_class: ast.ClassDef, name: str) -> ast.expr:
    for node in config_class.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return node.value
    raise AssertionError(f"Missing protected config assignment: {name}")


def _call_keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"Missing protected config keyword: {name}")


def _literal_keyword(call: ast.Call, name: str) -> Any:
    return ast.literal_eval(_call_keyword(call, name))


def _resolve_official_repo_root(repo_root: Path) -> Path:
    candidates = []
    if env_root := os.environ.get("MOLMOACT2_REPO_ROOT"):
        candidates.append(Path(env_root).expanduser())
    candidates.extend(parent / OFFICIAL_REPO_ARTIFACT_SUFFIX for parent in repo_root.parents)
    candidates.append(repo_root / OFFICIAL_REPO_ARTIFACT_SUFFIX)
    for candidate in candidates:
        if (candidate / OFFICIAL_REFERENCE_RELATIVE_DIR / "results.json").is_file():
            return candidate.resolve()
    checked = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Official successful MolmoAct2 YAM PNG reference run was not found. Checked:\n{checked}"
    )


def _image_statistics(image: np.ndarray) -> dict[str, Any]:
    rgb = image.astype(np.float32)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "channel_mean": rgb.mean(axis=(0, 1)).round(6).tolist(),
        "channel_std": rgb.std(axis=(0, 1)).round(6).tolist(),
        "luminance_percentiles": np.percentile(luminance, [5, 50, 95]).round(6).tolist(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_metric_image(image: Any, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise ValueError(
            f"Appearance metric {name} must have shape (360, 640, 3), got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"Appearance metric {name} must use uint8 pixels, got {array.dtype}")
    return array


def _luminance(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _tabletop_texture_coverage(luminance: np.ndarray) -> float:
    """Measure moderate local texture while tolerating renderer-specific shading.

    The fixed tabletop ROI excludes most borders and robot/background pixels. Moderate
    adjacent-pixel gradients expose the known unnaturally flat orange surface, while
    ignoring tiny antialiasing noise and strong object, shadow, or silhouette edges.
    """

    x_start = int(TABLETOP_ROI["x"][0] * IMAGE_SIZE[0])
    x_stop = int(TABLETOP_ROI["x"][1] * IMAGE_SIZE[0])
    y_start = int(TABLETOP_ROI["y"][0] * IMAGE_SIZE[1])
    y_stop = int(TABLETOP_ROI["y"][1] * IMAGE_SIZE[1])
    roi = luminance[y_start:y_stop, x_start:x_stop]
    horizontal = np.diff(roi, axis=1)[:-1]
    vertical = np.diff(roi, axis=0)[:, :-1]
    gradient_magnitude = np.hypot(horizontal, vertical)
    return float(np.mean((gradient_magnitude >= 2.0) & (gradient_magnitude <= 20.0)))


def compute_appearance_metrics(captured: Any, reference: Any) -> dict[str, float]:
    """Compare exact uint8 policy-input arrays without display conversion."""

    captured_array = _validate_metric_image(captured, name="captured image")
    reference_array = _validate_metric_image(reference, name="reference image")
    captured_rgb = captured_array.astype(np.float32)
    reference_rgb = reference_array.astype(np.float32)
    captured_mean = np.asarray(captured_rgb.mean(axis=(0, 1)).round(6).tolist())
    reference_mean = np.asarray(reference_rgb.mean(axis=(0, 1)).round(6).tolist())
    captured_luminance = _luminance(captured_array)
    reference_luminance = _luminance(reference_array)

    cdf_distances = []
    for channel in range(3):
        captured_histogram = np.bincount(
            captured_array[..., channel].ravel(), minlength=256
        ).astype(np.float64)
        reference_histogram = np.bincount(
            reference_array[..., channel].ravel(), minlength=256
        ).astype(np.float64)
        captured_cdf = np.cumsum(captured_histogram / captured_array[..., channel].size)
        reference_cdf = np.cumsum(reference_histogram / reference_array[..., channel].size)
        cdf_distances.append(float(np.abs(captured_cdf[:-1] - reference_cdf[:-1]).mean()))

    metrics = {
        "mean_rgb_l2_delta": float(np.linalg.norm(captured_mean - reference_mean)),
        "luminance_percentile_mae": float(
            np.mean(
                np.abs(
                    np.percentile(captured_luminance, LUMINANCE_PERCENTILES)
                    - np.percentile(reference_luminance, LUMINANCE_PERCENTILES)
                )
            )
        ),
        "rgb_cdf_distance": float(np.mean(cdf_distances)),
        "tabletop_texture_coverage_delta": abs(
            _tabletop_texture_coverage(captured_luminance)
            - _tabletop_texture_coverage(reference_luminance)
        ),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Appearance metrics must be finite")
    return metrics


def evaluate_metric_threshold(
    metric: str, observed_value: float, baseline_value: float
) -> dict[str, Any]:
    if metric not in APPEARANCE_THRESHOLDS:
        raise ValueError(f"Unknown appearance metric: {metric}")
    if not np.isfinite(observed_value) or not np.isfinite(baseline_value):
        raise ValueError("Appearance threshold values must be finite")
    threshold = APPEARANCE_THRESHOLDS[metric]
    absolute_limit = float(threshold["absolute_limit"])
    relative_limit = float(baseline_value * threshold["baseline_fraction"])
    absolute_pass = observed_value <= absolute_limit
    relative_pass = observed_value <= relative_limit
    reasons = []
    if not absolute_pass:
        reasons.append("absolute_limit_exceeded")
    if not relative_pass:
        reasons.append("relative_limit_exceeded")
    return {
        "status": "pass" if absolute_pass and relative_pass else "fail",
        "observed_value": float(observed_value),
        "absolute_limit": absolute_limit,
        "baseline_value": float(baseline_value),
        "relative_limit": relative_limit,
        "reason": "passed" if not reasons else ",".join(reasons),
    }


def _load_rgb_png(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".png":
        raise ValueError(f"Official visual reference must be PNG, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing official policy-input PNG: {path}")
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Official policy-input PNG must be RGB: {path} ({image.mode})")
        if image.size != IMAGE_SIZE:
            raise ValueError(
                f"Official policy-input PNG must be 640x360: {path} ({image.size[0]}x{image.size[1]})"
            )
        return np.asarray(image).copy()


def validate_request_frame_images(images: dict[str, Any]) -> dict[str, np.ndarray]:
    """Validate the exact ordered three-camera MolmoAct2 request-frame contract."""

    if not isinstance(images, dict):
        raise ValueError("Request frame must be an insertion-ordered camera dictionary")
    for name in CAMERA_ORDER:
        if name not in images:
            raise ValueError(f"Missing request-frame camera: {name}")
    actual_order = tuple(images)
    if actual_order != CAMERA_ORDER:
        raise ValueError(
            "Incorrect request-frame camera order: "
            f"expected {list(CAMERA_ORDER)}, got {list(actual_order)}"
        )
    validated: dict[str, np.ndarray] = {}
    for name in CAMERA_ORDER:
        image = images[name]
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Request-frame camera '{name}' must be a NumPy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Request-frame camera '{name}' must be RGB, got shape {image.shape}")
        if image.shape[:2] != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
            raise ValueError(
                f"Request-frame camera '{name}' must be 640x360, got "
                f"{image.shape[1]}x{image.shape[0]}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"Request-frame camera '{name}' must use uint8 pixels, got {image.dtype}"
            )
        validated[name] = image
    return validated


def validate_no_policy_evidence(runtime: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless every no-policy runtime counter is exact and explicit."""

    expected = {
        "policy_service_called": False,
        "policy_inference_called": False,
        "policy_actions_executed": False,
        "action_count": 0,
        "rollout_steps": 0,
    }
    failures = []
    for field, required in expected.items():
        observed = runtime.get(field)
        if isinstance(required, bool):
            passed = observed is required
        else:
            passed = type(observed) is int and observed == required
        if not passed:
            failures.append(
                {
                    "reason": "no_policy_evidence",
                    "field": field,
                    "required": required,
                    "observed": observed,
                }
            )
    return {
        "status": "fail" if failures else "pass",
        "required": expected,
        "observed": {field: runtime.get(field) for field in expected},
        "predicates": {
            field: not any(item["field"] == field for item in failures) for field in expected
        },
        "failures": failures,
    }


def _collect_scene_report(repo_root: Path) -> dict[str, Any]:
    scene_path = repo_root / SCENE_RELATIVE_PATH
    root = ET.parse(scene_path).getroot()
    if root.find(".//texture[@type='skybox']") is not None:
        raise AssertionError("Gradient skybox was reactivated")
    if root.find(".//texture[@builtin='checker']") is not None:
        raise AssertionError("Checker floor texture was reactivated")
    if root.find(".//haze") is not None:
        raise AssertionError("Blue haze was reactivated")

    headlight = root.find("./visual/headlight")
    if headlight is None:
        raise AssertionError("Active scene must define the aligned headlight")
    ambient = _parse_float_vector(headlight.get("ambient"), name="headlight ambient")
    if not np.allclose(ambient, [0.3, 0.3, 0.3], atol=0.02):
        raise AssertionError(f"Headlight ambient must remain approximately 0.3, got {ambient}")
    for attribute in ("diffuse", "specular"):
        value = _parse_float_vector(headlight.get(attribute), name=f"headlight {attribute}")
        if not np.allclose(value, [0.0, 0.0, 0.0], atol=1e-9):
            raise AssertionError(f"Divergent headlight {attribute} was reactivated: {value}")

    lights = []
    for light in root.findall("./worldbody/light"):
        if light.get("directional") != "true":
            raise AssertionError(f"Active scene light is not directional: {light.get('name')}")
        diffuse = _parse_float_vector(light.get("diffuse"), name="light diffuse")
        if not np.allclose(diffuse, diffuse[0], atol=1e-9) or not 0.0 < diffuse[0] <= 1.0:
            raise AssertionError(f"Active scene light is not white: {light.get('name')}={diffuse}")
        lights.append(light.get("name"))
    expected_lights = ["front_key_light", "overhead_fill_light"]
    if lights != expected_lights:
        raise AssertionError(f"Expected exactly two aligned directional lights, got {lights}")

    meshes = {}
    for mesh in root.findall("./asset/mesh"):
        source_path = (scene_path.parent / str(mesh.get("file"))).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Referenced scene asset does not resolve: {source_path}")
        meshes[str(mesh.get("name"))] = str(source_path)

    geoms = {str(geom.get("name")): geom for geom in root.findall(".//geom")}
    for name, expected in EXPECTED_PHYSICAL_GEOMS.items():
        geom = geoms.get(name)
        if geom is None:
            raise AssertionError(f"Protected physical geom is missing: {name}")
        actual = {attribute: geom.get(attribute) for attribute in expected}
        if actual != expected:
            raise AssertionError(
                f"Protected physical geom changed: {name}; expected {expected}, got {actual}"
            )
    overlay = geoms.get("official_yam_tabletop_visual")
    if overlay is None or overlay.get("contype") != "0" or overlay.get("conaffinity") != "0":
        raise AssertionError("Official tabletop visual overlay must remain collision-disabled")

    option = root.find("./option")
    expected_option = {
        "integrator": "implicit",
        "timestep": "0.002",
        "gravity": "0 0 -9.81",
    }
    actual_option = {
        name: option.get(name) if option is not None else None for name in expected_option
    }
    if actual_option != expected_option:
        raise AssertionError(f"Protected scene dynamics changed: {actual_option}")

    mujoco.MjModel.from_xml_path(str(scene_path))
    return {
        "path": str(scene_path),
        "headlight_ambient": ambient,
        "active_directional_lights": lights,
        "resolved_mesh_assets": meshes,
        "protected_physical_geoms": sorted(EXPECTED_PHYSICAL_GEOMS),
        "model_loaded": True,
    }


def _collect_protected_config_report(repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / CONFIG_RELATIVE_PATH
    module = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    config_class = _find_config_class(module)
    base_module = ast.parse(
        (repo_root / BASE_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"),
        filename=str(repo_root / BASE_CONFIG_RELATIVE_PATH),
    )
    camera_specs = None
    for node in base_module.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "OFFICIAL_YAM_CAMERA_SPECS":
                camera_specs = ast.literal_eval(node.value)
                break
    if camera_specs != EXPECTED_CAMERA_SPECS:
        raise AssertionError(
            "Protected official YAM camera geometry changed: "
            f"expected {EXPECTED_CAMERA_SPECS}, got {camera_specs}"
        )

    camera_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MolmoAct2OfficialSimEvalYamBoxCameraSystem"
    )
    camera_assignment = _class_assignment(camera_class, "cameras")
    if not isinstance(camera_assignment, ast.List):
        raise AssertionError("Protected camera list is not a literal ordered list")
    camera_names = []
    camera_sources = {}
    for camera_call in camera_assignment.elts:
        if not isinstance(camera_call, ast.Call):
            raise AssertionError("Protected camera entry is malformed")
        name = _literal_keyword(camera_call, "name")
        camera_names.append(name)
        camera_sources[name] = {
            "reference_body_names": _literal_keyword(camera_call, "reference_body_names"),
            "offset_source": ast.unparse(_call_keyword(camera_call, "camera_offset")),
            "quaternion_source": ast.unparse(_call_keyword(camera_call, "camera_quaternion")),
            "fov_source": ast.unparse(_call_keyword(camera_call, "fov")),
        }
    if tuple(camera_names) != CAMERA_ORDER:
        raise AssertionError(f"Protected camera order changed: {camera_names}")

    task_sampler_call = _class_assignment(config_class, "task_sampler_config")
    policy_call = _class_assignment(config_class, "policy_config")
    robot_call = _class_assignment(config_class, "robot_config")
    task_call = _class_assignment(config_class, "task_config")
    if not all(
        isinstance(call, ast.Call)
        for call in (task_sampler_call, policy_call, robot_call, task_call)
    ):
        raise AssertionError("Protected config blocks are malformed")

    camera_mapping = _literal_keyword(policy_call, "camera_mapping")
    expected_mapping = {name: name for name in CAMERA_ORDER}
    action_noise_call = _call_keyword(robot_call, "action_noise_config")
    if not isinstance(action_noise_call, ast.Call):
        raise AssertionError("Protected action-noise config is malformed")

    protected = {
        "seed": ast.literal_eval(_class_assignment(config_class, "seed")),
        "camera_order": camera_names,
        "camera_specs": camera_specs,
        "camera_sources": camera_sources,
        "camera_mapping": camera_mapping,
        "endpoint_url": _literal_keyword(policy_call, "endpoint_url"),
        "remote_config": _literal_keyword(policy_call, "remote_config"),
        "execution_mode": _literal_keyword(policy_call, "execution_mode"),
        "execution_command_hz": float(_literal_keyword(policy_call, "execution_command_hz")),
        "num_steps": int(_literal_keyword(policy_call, "num_steps")),
        "n_action_steps": _literal_keyword(policy_call, "n_action_steps"),
        "action_noise_enabled": _literal_keyword(action_noise_call, "enabled"),
        "scene_xml_paths": _literal_keyword(task_sampler_call, "scene_xml_paths"),
        "sim_settle_timesteps": int(_literal_keyword(task_sampler_call, "sim_settle_timesteps")),
        "randomize_lighting": _literal_keyword(task_sampler_call, "randomize_lighting"),
        "randomize_textures": _literal_keyword(task_sampler_call, "randomize_textures"),
        "randomize_dynamics": _literal_keyword(task_sampler_call, "randomize_dynamics"),
        "robot_xml_path": ast.unparse(_call_keyword(robot_call, "robot_xml_path")),
        "robot_view_factory": ast.unparse(_call_keyword(robot_call, "robot_view_factory")),
        "task_class": ast.unparse(_call_keyword(task_call, "task_cls")),
    }
    expected = {
        "seed": 0,
        "camera_mapping": expected_mapping,
        "endpoint_url": "http://127.0.0.1:8203/act",
        "remote_config": {
            "host": "127.0.0.1",
            "port": 8203,
            "path": "/act",
            "timeout": 60.0,
        },
        "execution_mode": "sim_eval_step",
        "execution_command_hz": 30.0,
        "num_steps": 30,
        "n_action_steps": None,
        "action_noise_enabled": False,
        "scene_xml_paths": [SCENE_RELATIVE_PATH.as_posix()],
        "sim_settle_timesteps": 0,
        "randomize_lighting": False,
        "randomize_textures": False,
        "randomize_dynamics": False,
        "robot_xml_path": "Path('bimanual_yam_linear_flattened.xml')",
        "robot_view_factory": "OfficialSimEvalBimanualYamRobotView",
        "task_class": "MolmoAct2OfficialSimEvalYamBoxTask",
    }
    for name, expected_value in expected.items():
        if protected[name] != expected_value:
            raise AssertionError(
                f"Protected camera/action/control/timing/task value changed: "
                f"{name} expected {expected_value!r}, got {protected[name]!r}"
            )
    return protected


def _collect_official_reference_report(repo_root: Path) -> dict[str, Any]:
    official_root = _resolve_official_repo_root(repo_root)
    run_dir = official_root / OFFICIAL_REFERENCE_RELATIVE_DIR
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    task_results = results["tasks"]["BimanualYAMPutEverythingInBox-v1"]
    episode = task_results["episodes"][0]
    if episode.get("success") is not True or task_results.get("success_rate") != 1.0:
        raise AssertionError(f"Official PNG reference run is not successful: {results_path}")

    images = {}
    for name in CAMERA_ORDER:
        path = official_root / OFFICIAL_REFERENCE_FRAME_DIR / f"ep000_{name}.png"
        image = _load_rgb_png(path)
        digest = _sha256(path)
        if digest != OFFICIAL_REFERENCE_SHA256[name]:
            raise AssertionError(
                f"Official policy-input PNG hash mismatch for {name}: "
                f"expected {OFFICIAL_REFERENCE_SHA256[name]}, got {digest}"
            )
        images[name] = {
            "path": str(path),
            "suffix": path.suffix.lower(),
            "sha256": digest,
            "statistics": _image_statistics(image),
        }
    return {
        "repository_root": str(official_root),
        "relative_directory": OFFICIAL_REFERENCE_RELATIVE_DIR.as_posix(),
        "results_path": str(results_path),
        "results": {
            "episode_success": bool(episode["success"]),
            "success_rate": float(task_results["success_rate"]),
            "steps": int(episode["steps"]),
        },
        "images": images,
        "human_video_used": False,
    }


def _collect_rejected_baseline_report(
    repo_root: Path, reference_report: dict[str, Any]
) -> dict[str, Any]:
    run_dir = repo_root / REJECTED_BASELINE_RELATIVE_DIR
    images = {}
    captured = {}
    for name in CAMERA_ORDER:
        path = run_dir / f"{name}.png"
        image = _load_rgb_png(path)
        digest = _sha256(path)
        if digest != REJECTED_BASELINE_SHA256[name]:
            raise AssertionError(
                f"Rejected baseline PNG hash mismatch for {name}: "
                f"expected {REJECTED_BASELINE_SHA256[name]}, got {digest}"
            )
        captured[name] = image
        images[name] = {"path": str(path), "suffix": path.suffix.lower(), "sha256": digest}

    comparison = build_appearance_comparison(captured, reference_report)
    for name in CAMERA_ORDER:
        for metric, pinned_value in REJECTED_BASELINE_METRICS[name].items():
            observed = comparison[name]["metrics"][metric]
            tolerance = 1e-12 if metric == "mean_rgb_l2_delta" else 5e-6
            if not np.isclose(observed, pinned_value, rtol=0.0, atol=tolerance):
                raise AssertionError(
                    f"Rejected baseline metric mismatch for {name}/{metric}: "
                    f"expected {pinned_value}, got {observed}"
                )
    return {
        "relative_directory": REJECTED_BASELINE_RELATIVE_DIR.as_posix(),
        "images": images,
        "metrics": REJECTED_BASELINE_METRICS,
    }


def collect_static_regression_report(repo_root: Path | None = None) -> dict[str, Any]:
    """Run focused static visual and protected-contract checks."""

    repo_root = (repo_root or _repo_root()).resolve()
    scene = _collect_scene_report(repo_root)
    protected_contract = _collect_protected_config_report(repo_root)
    official_references = _collect_official_reference_report(repo_root)
    rejected_baseline = _collect_rejected_baseline_report(repo_root, official_references)
    contract_pass = bool(scene["model_loaded"] and protected_contract)
    reference_pass = bool(
        official_references["results"]["episode_success"]
        and official_references["results"]["success_rate"] == 1.0
        and all(
            official_references["images"][name]["sha256"] == OFFICIAL_REFERENCE_SHA256[name]
            for name in CAMERA_ORDER
        )
    )
    return {
        "status": "pass" if contract_pass and reference_pass else "fail",
        "scene": scene,
        "protected_contract": protected_contract,
        "official_references": official_references,
        "rejected_baseline": rejected_baseline,
    }


def build_appearance_comparison(
    captured: dict[str, np.ndarray], reference_report: dict[str, Any]
) -> dict[str, Any]:
    captured = validate_request_frame_images(captured)
    comparisons = {}
    for name in CAMERA_ORDER:
        reference_path = Path(reference_report["images"][name]["path"])
        reference = _load_rgb_png(reference_path)
        metrics = compute_appearance_metrics(captured[name], reference)
        predicates = {
            metric: evaluate_metric_threshold(
                metric, observed, REJECTED_BASELINE_METRICS[name][metric]
            )
            for metric, observed in metrics.items()
        }
        comparisons[name] = {
            "status": (
                "pass"
                if all(predicate["status"] == "pass" for predicate in predicates.values())
                else "fail"
            ),
            "captured": _image_statistics(captured[name]),
            "official_reference": _image_statistics(reference),
            "metrics": metrics,
            "predicates": predicates,
        }
    return comparisons


def derive_report_status(
    *,
    contract_pass: bool,
    no_policy_pass: bool,
    reference_pass: bool,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    for reason, passed in (
        ("contract", contract_pass),
        ("no_policy", no_policy_pass),
        ("reference", reference_pass),
    ):
        if not passed:
            failures.append({"reason": reason})
    for camera in CAMERA_ORDER:
        for metric, predicate in comparison[camera]["predicates"].items():
            if predicate["status"] != "pass":
                failures.append(
                    {
                        "metric": metric,
                        "camera": camera,
                        "observed_value": predicate["observed_value"],
                        "absolute_limit": predicate["absolute_limit"],
                        "baseline_value": predicate["baseline_value"],
                        "relative_limit": predicate["relative_limit"],
                        "reason": predicate["reason"],
                    }
                )
    return {"status": "fail" if failures else "pass", "failures": failures}


def capture_request_frames(
    repo_root: Path | None = None,
    seed: int = SCENE_SEED,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Initialize one active task and build one request without calling ``/act``."""

    from molmo_spaces.data_generation.config.molmoact2_official_sim_eval_yam_box_config import (
        MolmoAct2OfficialSimEvalYamBoxDataGenConfig,
    )
    from molmo_spaces.env import env as env_module
    from molmo_spaces.policy.learned_policy.molmoact2_yam_policy import (
        build_molmoact2_yam_request,
    )
    from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer

    repo_root = (repo_root or _repo_root()).resolve()
    config = MolmoAct2OfficialSimEvalYamBoxDataGenConfig()
    config.seed = seed
    original_renderer = env_module.MjOpenGLRenderer

    class SmokeEGLRenderer(MjOpenGLRenderer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["device_id"] = 0
            super().__init__(*args, **kwargs)

    env_module.MjOpenGLRenderer = SmokeEGLRenderer
    sampler = None
    task = None
    try:
        sampler = config.task_sampler_config.task_sampler_class(config)
        task = sampler.sample_task(force_advance_scene=True, house_index=0)
        if task is None:
            raise RuntimeError("Active YAM scene initialization returned no task")
        observations = task.get_observations()
        request = build_molmoact2_yam_request(
            observations,
            instruction=config.policy_config.instruction_override,
            camera_mapping=config.policy_config.camera_mapping,
            gripper_max=config.policy_config.gripper_max,
            gripper_open_command=config.policy_config.gripper_open_command,
            gripper_closed_command=config.policy_config.gripper_closed_command,
        )
        images = validate_request_frame_images({name: request[name] for name in CAMERA_ORDER})
        metadata = {
            "policy_service_called": False,
            "policy_inference_called": False,
            "policy_actions_executed": False,
            "action_count": 0,
            "rollout_steps": 0,
            "seed": config.seed,
            "renderer_backend": "egl-device-0",
            "instruction": request["instruction"],
            "state_shape": list(np.asarray(request["state"]).shape),
            "camera_order": list(images),
            "source_scene": str(repo_root / SCENE_RELATIVE_PATH),
        }
        return images, metadata
    finally:
        if task is not None:
            task.close()
            task.__dict__.clear()
            task = None
        if sampler is not None:
            sampler.close()
            sampler = None
        env_module.MjOpenGLRenderer = original_renderer
        gc.collect()


def run_smoke_gate(repo_root: Path | None = None, seed: int = SCENE_SEED) -> Path:
    """Capture and compare one three-camera request frame into a timestamped run."""

    repo_root = (repo_root or _repo_root()).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = repo_root / RUN_ARTIFACT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    static_report: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    captured: dict[str, np.ndarray] | None = None
    comparison: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    try:
        static_report = collect_static_regression_report(repo_root)
    except Exception as exc:
        failures.append(
            {
                "reason": "static_contract_error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    try:
        raw_captured, runtime = capture_request_frames(repo_root, seed)
    except Exception as exc:
        failures.append(
            {
                "reason": "capture_error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    else:
        try:
            captured = validate_request_frame_images(raw_captured)
        except ValueError as exc:
            failures.append(
                {
                    "reason": "malformed_request_frame",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    if captured is not None and static_report.get("official_references"):
        try:
            comparison = build_appearance_comparison(captured, static_report["official_references"])
        except Exception as exc:
            failures.append(
                {
                    "reason": "reference_comparison_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    artifact_images = captured or {
        name: np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8) for name in CAMERA_ORDER
    }
    image_report = {}
    artifact_files = {}
    for name in CAMERA_ORDER:
        filename = f"{name}.png"
        output_path = output_dir / filename
        Image.fromarray(artifact_images[name]).save(output_path, format="PNG")
        artifact_hash = _sha256(output_path)
        image_report[name] = {
            "path": str(output_path.relative_to(repo_root)),
            "sha256": artifact_hash,
            "statistics": _image_statistics(artifact_images[name]),
            "diagnostic_placeholder": captured is None,
        }
        artifact_files[filename] = artifact_hash

    no_policy = validate_no_policy_evidence(runtime)
    failures.extend(no_policy["failures"])
    contract_pass = static_report.get("status") == "pass"
    if not contract_pass:
        failures.append({"reason": "contract"})
    references = static_report.get("official_references", {})
    reference_results = references.get("results", {})
    reference_pass = bool(
        reference_results.get("episode_success") is True
        and reference_results.get("success_rate") == 1.0
    )
    if not reference_pass:
        failures.append({"reason": "reference"})
    if comparison:
        metric_status = derive_report_status(
            contract_pass=True,
            no_policy_pass=True,
            reference_pass=True,
            comparison=comparison,
        )
        failures.extend(metric_status["failures"])

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "fail" if failures else "pass",
        "failures": failures,
        "gate": "no-policy three-camera MolmoAct2 YAM request frame",
        "timestamp_utc": timestamp,
        "camera_order": list(CAMERA_ORDER),
        "ordered_cameras": list(CAMERA_ORDER),
        "seed": runtime.get("seed", SCENE_SEED),
        "runtime": runtime,
        "no_policy": no_policy,
        "captured_images": image_report,
        "official_references": references,
        "rejected_baseline": static_report.get("rejected_baseline", {}),
        "appearance_contract": {
            "input": "exact ordered 640x360 uint8 RGB policy-input arrays",
            "luminance": "0.2126R+0.7152G+0.0722B",
            "luminance_percentiles": list(LUMINANCE_PERCENTILES),
            "rgb_histogram_bins": 256,
            "rgb_cdf_intervals": 255,
            "tabletop_roi": {key: list(value) for key, value in TABLETOP_ROI.items()},
            "tabletop_gradient_range_inclusive": [2.0, 20.0],
            "tabletop_texture_rationale": (
                "Moderate adjacent-pixel luminance gradients detect the known flat "
                "orange tabletop while tolerating non-identical renderer shading, "
                "antialiasing noise, and strong object or silhouette edges."
            ),
            "thresholds": APPEARANCE_THRESHOLDS,
            "rejected_baseline": {
                "relative_directory": REJECTED_BASELINE_RELATIVE_DIR.as_posix(),
                "sha256": REJECTED_BASELINE_SHA256,
                "metrics": REJECTED_BASELINE_METRICS,
            },
        },
        "comparison": comparison,
        "write_scope": str(output_dir.relative_to(repo_root)),
        "artifact_files": artifact_files,
        "human_video_used": False,
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def validate_report(report_path: Path) -> dict[str, Any]:
    """Independently validate one completed fixed-contract artifact report."""

    report_path = report_path.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError("Report status must be pass")
    if report.get("camera_order") != list(CAMERA_ORDER):
        raise ValueError("Report camera_order does not match the protected order")
    if report.get("seed") != 0 or report.get("runtime", {}).get("seed") != 0:
        raise ValueError("Report and runtime seed must both be 0")

    no_policy = validate_no_policy_evidence(report.get("runtime", {}))
    if no_policy["status"] != "pass":
        raise ValueError(f"Report no-policy evidence failed: {no_policy['failures']}")

    expected_files = {"comparison_report.json", *(f"{name}.png" for name in CAMERA_ORDER)}
    observed_files = {path.name for path in report_path.parent.iterdir() if path.is_file()}
    if observed_files != expected_files:
        raise ValueError(
            f"Artifact directory must contain exactly {sorted(expected_files)}, "
            f"observed {sorted(observed_files)}"
        )
    artifact_files = report.get("artifact_files")
    expected_image_files = {f"{name}.png" for name in CAMERA_ORDER}
    if not isinstance(artifact_files, dict) or set(artifact_files) != expected_image_files:
        raise ValueError("Report artifact_files must declare exactly the three camera PNGs")
    for name in CAMERA_ORDER:
        filename = f"{name}.png"
        image_path = report_path.parent / filename
        if _sha256(image_path) != artifact_files[filename]:
            raise ValueError(f"Artifact hash mismatch: {filename}")
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != IMAGE_SIZE:
                raise ValueError(
                    f"Artifact {filename} must be a 640x360 RGB PNG, observed "
                    f"format={image.format}, mode={image.mode}, size={image.size}"
                )
            array = np.asarray(image)
        if array.dtype != np.uint8 or array.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
            raise ValueError(f"Artifact {filename} must decode to uint8 640x360 RGB")

    comparison = report.get("comparison")
    if not isinstance(comparison, dict) or set(comparison) != set(CAMERA_ORDER):
        raise ValueError("Report comparison must contain exactly the protected cameras")
    if any(comparison[name].get("status") != "pass" for name in CAMERA_ORDER):
        raise ValueError("Every camera comparison status must be pass")

    references = report.get("official_references", {})
    results = references.get("results", {})
    if results.get("episode_success") is not True or results.get("success_rate") != 1.0:
        raise ValueError("Official reference results must prove a successful episode")
    reference_images = references.get("images")
    if not isinstance(reference_images, dict) or set(reference_images) != set(CAMERA_ORDER):
        raise ValueError("Official references must contain exactly the protected cameras")
    for name in CAMERA_ORDER:
        item = reference_images[name]
        path = Path(item.get("path", ""))
        if path.name != f"ep000_{name}.png" or path.suffix.lower() != ".png":
            raise ValueError(f"Official reference for {name} must be ep000_{name}.png")
        if "mp4" in json.dumps(item).lower():
            raise ValueError("Official appearance references must not mention MP4 files")
    if report.get("human_video_used") is not False:
        raise ValueError("Report must explicitly prove no human video was used")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run focused static checks without initializing the scene renderer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SCENE_SEED,
        help=f"Deterministic task seed for request-frame capture (default: {SCENE_SEED}).",
    )
    parser.add_argument(
        "--validate-report",
        type=Path,
        metavar="REPORT",
        help="Validate an existing artifact report without initializing or capturing.",
    )
    args = parser.parse_args()
    if args.validate_report is not None:
        validate_report(args.validate_report)
        print(args.validate_report)
        return
    if args.static_only:
        print(json.dumps(collect_static_regression_report(), indent=2, sort_keys=True))
        return
    report_path = run_smoke_gate(seed=args.seed)
    print(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
