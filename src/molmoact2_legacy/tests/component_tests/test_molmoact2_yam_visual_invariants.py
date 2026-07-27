"""Independent physical and camera/control fingerprints for MolmoAct2 YAM."""

from __future__ import annotations

import ast
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "examples/molmoact2_official_sim_eval_yam_box/scene.xml"
CONFIG_PATH = REPO_ROOT / (
    "molmo_spaces/data_generation/config/molmoact2_official_sim_eval_yam_box_config.py"
)
BASE_CONFIG_PATH = REPO_ROOT / (
    "molmo_spaces/data_generation/config/molmoact2_official_yam_box_config.py"
)
POLICY_PATH = REPO_ROOT / "molmo_spaces/policy/learned_policy/molmoact2_yam_policy.py"
ROBOT_VIEW_PATH = REPO_ROOT / (
    "molmo_spaces/robots/robot_views/official_sim_eval_bimanual_yam_view.py"
)
FIXTURE_ROOT = Path(__file__).with_name("fixtures")
PHYSICAL_FIXTURE_PATH = FIXTURE_ROOT / "molmoact2_yam_pre_visual_physics.json"
CAMERA_CONTROL_FIXTURE_PATH = FIXTURE_ROOT / "molmoact2_yam_camera_control.json"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _module(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _assignment(module: ast.Module, name: str) -> ast.expr:
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return node.value
    raise AssertionError(f"missing assignment: {name}")


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name
    )


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _ast_signature(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _class_assignment(class_node: ast.ClassDef, name: str) -> ast.expr:
    for node in class_node.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return node.value
    raise AssertionError(f"missing class assignment: {name}")


def _literal(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def _resolved_literal(module: ast.Module, node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        return _resolved_literal(module, _assignment(module, node.id))
    if isinstance(node, ast.Dict):
        return {
            _resolved_literal(module, key): _resolved_literal(module, value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_resolved_literal(module, element) for element in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    return _literal(node)


def _rounded(values: Any) -> Any:
    array = np.asarray(values)
    if array.ndim == 0:
        return round(float(array), 12)
    return np.round(array.astype(float), 12).tolist()


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or f"unnamed:{index}"


def _physical_fingerprint(scene_path: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    visual_names = {
        geom.get("name")
        for geom in ET.parse(scene_path).getroot().findall(".//geom[@class='visual']")
    }
    physical_geom_ids = [
        index
        for index in range(model.ngeom)
        if _name(model, mujoco.mjtObj.mjOBJ_GEOM, index) not in visual_names
    ]
    root = ET.parse(scene_path).getroot()
    contacts = [
        {"tag": element.tag, **dict(sorted(element.attrib.items()))}
        for element in root.findall("./contact/*")
    ]
    return {
        "schema_version": 1,
        "option": {
            "integrator": int(model.opt.integrator),
            "solver": int(model.opt.solver),
            "cone": int(model.opt.cone),
            "jacobian": int(model.opt.jacobian),
            "timestep": _rounded(model.opt.timestep),
            "gravity": _rounded(model.opt.gravity),
            "iterations": int(model.opt.iterations),
            "tolerance": _rounded(model.opt.tolerance),
        },
        "bodies": [
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, index),
                "parent": int(model.body_parentid[index]),
                "pos": _rounded(model.body_pos[index]),
                "quat": _rounded(model.body_quat[index]),
                "mass": _rounded(model.body_mass[index]),
                "inertia": _rounded(model.body_inertia[index]),
                "ipos": _rounded(model.body_ipos[index]),
                "iquat": _rounded(model.body_iquat[index]),
            }
            for index in range(model.nbody)
        ],
        "joints": [
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, index),
                "body": int(model.jnt_bodyid[index]),
                "type": int(model.jnt_type[index]),
                "pos": _rounded(model.jnt_pos[index]),
                "axis": _rounded(model.jnt_axis[index]),
                "range": _rounded(model.jnt_range[index]),
                "limited": int(model.jnt_limited[index]),
                "damping": _rounded(model.dof_damping[model.jnt_dofadr[index]]),
                "frictionloss": _rounded(model.dof_frictionloss[model.jnt_dofadr[index]]),
                "armature": _rounded(model.dof_armature[model.jnt_dofadr[index]]),
            }
            for index in range(model.njnt)
        ],
        "actuators": [
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index),
                "trntype": int(model.actuator_trntype[index]),
                "trnid": model.actuator_trnid[index].tolist(),
                "ctrlrange": _rounded(model.actuator_ctrlrange[index]),
                "forcerange": _rounded(model.actuator_forcerange[index]),
                "gear": _rounded(model.actuator_gear[index]),
            }
            for index in range(model.nu)
        ],
        "physical_geoms": [
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, index),
                "body": int(model.geom_bodyid[index]),
                "type": int(model.geom_type[index]),
                "pos": _rounded(model.geom_pos[index]),
                "quat": _rounded(model.geom_quat[index]),
                "size": _rounded(model.geom_size[index]),
                "contype": int(model.geom_contype[index]),
                "conaffinity": int(model.geom_conaffinity[index]),
                "friction": _rounded(model.geom_friction[index]),
                "solref": _rounded(model.geom_solref[index]),
                "solimp": _rounded(model.geom_solimp[index]),
            }
            for index in physical_geom_ids
        ],
        "contacts": contacts,
        "task_geometry": {
            "object_bodies": [
                "obj_073-a_lego_duplo",
                "obj_056_tennis_ball",
                "open_box",
            ],
            "authoritative_geom_names": [
                _name(model, mujoco.mjtObj.mjOBJ_GEOM, index) for index in physical_geom_ids
            ],
        },
    }


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _call_keywords(call: ast.Call) -> dict[str, ast.expr]:
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


def _camera_spec_key(node: ast.AST) -> str:
    assert isinstance(node, ast.Subscript)
    assert isinstance(node.value, ast.Subscript)
    return _literal(node.value.slice)


def _molmospaces_quaternion(quat_wxyz: list[float]) -> list[float]:
    sapien_rot = Rotation.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    axes = np.column_stack(([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]))
    return _rounded(Rotation.from_matrix(sapien_rot @ axes).as_quat(scalar_first=True))


def _action_mapping(policy_module: ast.Module) -> dict[str, Any]:
    function = _function(policy_module, "molmoact2_yam_action_to_move_group_command")
    return_node = next(node for node in ast.walk(function) if isinstance(node, ast.Return))
    assert isinstance(return_node.value, ast.Dict)
    mapping = {}
    for key_node, value_node in zip(return_node.value.keys, return_node.value.values, strict=True):
        key = _literal(key_node)
        target = value_node
        if isinstance(target, ast.Call) and isinstance(target.func, ast.Attribute):
            target = target.func.value
        if isinstance(target, ast.Call):
            target = target.args[0]
        if isinstance(target, ast.Call):
            target = target.args[0]
        assert isinstance(target, ast.Subscript)
        if isinstance(target.slice, ast.Slice):
            mapping[key] = [_literal(target.slice.lower), _literal(target.slice.upper)]
        else:
            mapping[key] = _literal(target.slice)
    return mapping


def _camera_control_fingerprint(
    scene_path: Path = SCENE_PATH,
    config_path: Path = CONFIG_PATH,
    base_config_path: Path = BASE_CONFIG_PATH,
    policy_path: Path = POLICY_PATH,
    robot_view_path: Path = ROBOT_VIEW_PATH,
) -> dict[str, Any]:
    base_module = _module(base_config_path)
    config_module = _module(config_path)
    policy_module = _module(policy_path)
    robot_module = _module(robot_view_path)
    camera_specs = _literal(_assignment(base_module, "OFFICIAL_YAM_CAMERA_SPECS"))
    camera_class = _class(config_module, "MolmoAct2OfficialSimEvalYamBoxCameraSystem")
    resolution = list(_literal(_class_assignment(camera_class, "img_resolution")))
    camera_calls = _literal_camera_calls(_class_assignment(camera_class, "cameras"))
    cameras = []
    for call in camera_calls:
        keywords = _call_keywords(call)
        spec_key = _camera_spec_key(keywords["camera_offset"])
        spec = camera_specs[spec_key]
        hfov = float(spec["hfov_deg"])
        width, height = spec["resolution"]
        vfov = math.degrees(2.0 * math.atan((height / width) * math.tan(math.radians(hfov / 2.0))))
        cameras.append(
            {
                "name": _literal(keywords["name"]),
                "source_spec": spec_key,
                "reference_bodies": _literal(keywords["reference_body_names"]),
                "offset": spec["p"],
                "source_quaternion_wxyz": spec["q"],
                "molmospaces_quaternion_wxyz": _molmospaces_quaternion(spec["q"]),
                "hfov_deg": hfov,
                "vfov_deg": round(vfov, 12),
                "resolution": list(spec["resolution"]),
            }
        )

    config_class = _class(config_module, "MolmoAct2OfficialSimEvalYamBoxDataGenConfig")
    robot_call = _class_assignment(config_class, "robot_config")
    sampler_call = _class_assignment(config_class, "task_sampler_config")
    policy_call = _class_assignment(config_class, "policy_config")
    assert isinstance(robot_call, ast.Call)
    assert isinstance(sampler_call, ast.Call)
    assert isinstance(policy_call, ast.Call)
    robot_keywords = _call_keywords(robot_call)
    sampler_keywords = _call_keywords(sampler_call)
    policy_keywords = _call_keywords(policy_call)
    init_qpos = _literal(robot_keywords["init_qpos"])
    visual_map = ET.parse(scene_path).getroot().find("./visual/map")
    assert visual_map is not None
    policy_source = _source(policy_path)
    return {
        "schema_version": 1,
        "camera_order": [camera["name"] for camera in cameras],
        "cameras": cameras,
        "projection": {
            "near": float(visual_map.get("znear")),
            "far": float(visual_map.get("zfar")),
            "intrinsics": "pinhole-from-horizontal-fov",
        },
        "image_semantics": {
            "output_resolution": resolution,
            "layout": "HWC",
            "color": "RGB",
            "dtype": "uint8",
            "batch_axis": "remove-only-singleton-leading-axis",
            "alpha": "drop-fourth-channel",
            "crop": "none",
            "resize": "none",
            "flip": "none",
            "channel_reorder": "none",
            "source_checks": {
                "accepts_hwc_3_or_4": "image.shape[-1] not in (3, 4)" in policy_source,
                "drops_alpha": "image = image[..., :3]" in policy_source,
                "scales_unit_float": "scale = 255.0" in policy_source,
            },
            "adapter_ast": _ast_signature(_function(policy_module, "_to_uint8_image")),
        },
        "action_bridge": {
            "state_action_order": list(
                _literal(_assignment(policy_module, "MOLMOACT2_YAM_STATE_ACTION_ORDER"))
            ),
            "action_dimension": _literal(_assignment(policy_module, "MOLMOACT2_YAM_ACTION_DIM")),
            "mapping": _action_mapping(policy_module),
            "gripper_semantics": _literal(
                _assignment(policy_module, "MOLMOACT2_YAM_ACTION_GRIPPER_SEMANTICS")
            ),
            "gripper_open_command": _literal(
                _assignment(robot_module, "OFFICIAL_YAM_GRIPPER_OPEN_COMMAND")
            ),
            "gripper_closed_command": _literal(
                _assignment(robot_module, "OFFICIAL_YAM_GRIPPER_CLOSED_COMMAND")
            ),
            "bridge_ast": _ast_signature(
                _function(policy_module, "molmoact2_yam_action_to_move_group_command")
            ),
        },
        "execution": {
            "mode": _literal(policy_keywords["execution_mode"]),
            "execution_command_hz": _literal(policy_keywords["execution_command_hz"]),
            "num_steps": _literal(policy_keywords["num_steps"]),
            "camera_mapping": _literal(policy_keywords["camera_mapping"]),
            "sim_settle_timesteps": _literal(sampler_keywords["sim_settle_timesteps"]),
            "randomize_lighting": _literal(sampler_keywords["randomize_lighting"]),
            "randomize_textures": _literal(sampler_keywords["randomize_textures"]),
            "randomize_dynamics": _literal(sampler_keywords["randomize_dynamics"]),
        },
        "initialization": {
            "seed": _literal(_class_assignment(config_class, "seed")),
            "init_qpos": init_qpos,
            "init_qpos_noise_range": _literal(robot_keywords["init_qpos_noise_range"]),
            "robot_world_pos": list(
                _literal(_assignment(base_module, "OFFICIAL_YAM_ROBOT_WORLD_POS"))
            ),
            "object_anchors_xy": _resolved_literal(
                base_module, _assignment(base_module, "OFFICIAL_YAM_OBJECT_ANCHORS_XY")
            ),
        },
        "instruction": _literal(_assignment(base_module, "OFFICIAL_MOLMOACT2_YAM_INSTRUCTION")),
        "success": {
            "object_names": list(
                _resolved_literal(
                    base_module, _assignment(base_module, "OFFICIAL_YAM_OBJECT_NAMES")
                )
            ),
            "box_center_xy": list(_literal(_assignment(base_module, "OFFICIAL_YAM_BOX_POS_XY"))),
            "inside_xy": "abs(x-box_x) < box_inner_half and abs(y-box_y) < box_inner_half",
            "box_inner_half": _literal(_assignment(base_module, "OFFICIAL_YAM_BOX_INNER_HALF")),
            "floor_top_z": 0.008,
            "z_lower_margin": 0.01,
            "rim_z": 0.068,
            "z_upper_margin": 0.05,
            "aggregate": "all-listed-objects-in-box",
            "predicate_ast": _ast_signature(
                _function(config_module, "official_sim_eval_yam_box_success_from_positions")
            ),
        },
    }


def _literal_camera_calls(node: ast.expr) -> list[ast.Call]:
    assert isinstance(node, ast.List)
    assert all(isinstance(element, ast.Call) for element in node.elts)
    return list(node.elts)


def _fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_fixture(actual: dict[str, Any], fixture_path: Path) -> None:
    assert _canonical_bytes(actual) == fixture_path.read_bytes()


def test_physical_model_matches_protected_pre_visual_fingerprint() -> None:
    _assert_fixture(_physical_fingerprint(SCENE_PATH), PHYSICAL_FIXTURE_PATH)


def test_camera_control_matches_protected_fingerprint() -> None:
    _assert_fixture(_camera_control_fingerprint(), CAMERA_CONTROL_FIXTURE_PATH)


def test_physical_fingerprint_detects_changed_friction_and_mass(tmp_path: Path) -> None:
    copied_scene = tmp_path / "scene"
    shutil.copytree(SCENE_PATH.parent, copied_scene)
    scene_path = copied_scene / "scene.xml"
    root = ET.parse(scene_path).getroot()
    geom = root.find(".//geom[@name='obj_056_tennis_ball_geom']")
    assert geom is not None
    geom.set("friction", "0.5 0.01 0.001")
    geom.set("mass", "0.1")
    ET.ElementTree(root).write(scene_path, encoding="unicode")

    with pytest.raises(AssertionError):
        _assert_fixture(_physical_fingerprint(scene_path), PHYSICAL_FIXTURE_PATH)


@pytest.mark.parametrize("field", ["offset", "hfov_deg"])
def test_camera_fingerprint_detects_transform_or_fov_change(field: str) -> None:
    fingerprint = _camera_control_fingerprint()
    fingerprint["cameras"][0][field] = [9.0, 9.0, 9.0] if field == "offset" else 10.0

    with pytest.raises(AssertionError):
        _assert_fixture(fingerprint, CAMERA_CONTROL_FIXTURE_PATH)


def test_camera_fingerprint_detects_order_change() -> None:
    fingerprint = _camera_control_fingerprint()
    fingerprint["camera_order"][0], fingerprint["camera_order"][1] = (
        fingerprint["camera_order"][1],
        fingerprint["camera_order"][0],
    )

    with pytest.raises(AssertionError):
        _assert_fixture(fingerprint, CAMERA_CONTROL_FIXTURE_PATH)


@pytest.mark.parametrize(
    ("section", "field", "changed"),
    [
        ("execution", "execution_command_hz", 20.0),
        ("action_bridge", "mapping", {"left_arm": [1, 7]}),
    ],
)
def test_control_fingerprint_detects_frequency_or_action_mapping_change(
    section: str, field: str, changed: Any
) -> None:
    fingerprint = _camera_control_fingerprint()
    fingerprint[section][field] = changed

    with pytest.raises(AssertionError):
        _assert_fixture(fingerprint, CAMERA_CONTROL_FIXTURE_PATH)


def test_fixture_comparison_is_non_destructive() -> None:
    before = {
        PHYSICAL_FIXTURE_PATH: PHYSICAL_FIXTURE_PATH.read_bytes(),
        CAMERA_CONTROL_FIXTURE_PATH: CAMERA_CONTROL_FIXTURE_PATH.read_bytes(),
    }

    _physical_fingerprint(SCENE_PATH)
    _camera_control_fingerprint()

    assert {path: path.read_bytes() for path in before} == before
