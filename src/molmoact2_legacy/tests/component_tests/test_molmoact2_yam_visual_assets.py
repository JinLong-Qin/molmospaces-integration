"""Provenance checks for vendored MolmoAct2 YAM visual sources."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import mujoco
import numpy as np
import pytest
import trimesh
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPO_ROOT / "examples/molmoact2_official_sim_eval_yam_box/assets"
MANIFEST_PATH = ASSET_ROOT / "visual_asset_provenance.json"
SCENE_PATH = REPO_ROOT / "examples/molmoact2_official_sim_eval_yam_box/scene.xml"
CONVERSION_SCRIPT = REPO_ROOT / "scripts/convert_molmoact2_yam_visual_assets.py"

EXPECTED_SOURCES = {
    "sources/maniskill/table/table.glb": {
        "bytes": 3_891_588,
        "sha256": "cb0ebd8ad6438c1160f095d902bf55415f9d643f8c6990a186a052251fe7951a",
    },
    "sources/mani_skill2_ycb/056_tennis_ball/textured.obj": {
        "bytes": 1_026_230,
        "sha256": "f00b576eaa1fb880d03f63be175836d1b5dbb1e9db95fd7b08c885984861a3d1",
    },
    "sources/mani_skill2_ycb/056_tennis_ball/textured.mtl": {
        "bytes": 63,
        "sha256": "24df6d9fd8366665821941bde13d9e4226d570193608ba4c9e4111b2908b8ec5",
    },
    "sources/mani_skill2_ycb/056_tennis_ball/texture_map.png": {
        "bytes": 3_503,
        "sha256": "4b2ef3ca763853d8343c3021a12064a301b29b6a233692743919cff144e16a71",
    },
    "sources/mani_skill2_ycb/073-a_lego_duplo/textured.obj": {
        "bytes": 1_063_498,
        "sha256": "8cac26820dcd0d0e8a1a74594e1abb2a7617fcdc52085fea94038ea67f4fe4a4",
    },
    "sources/mani_skill2_ycb/073-a_lego_duplo/textured.mtl": {
        "bytes": 63,
        "sha256": "24df6d9fd8366665821941bde13d9e4226d570193608ba4c9e4111b2908b8ec5",
    },
    "sources/mani_skill2_ycb/073-a_lego_duplo/texture_map.png": {
        "bytes": 4_905,
        "sha256": "018223f2398dee464e5ef07c937488016a9facdc1c234030ad7bc5c3ad30bdf9",
    },
}


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_asset_path(asset_root: Path, value: str) -> Path:
    parsed = urlparse(value)
    relative_path = PurePosixPath(value)
    assert parsed.scheme == "", f"network-backed asset path: {value}"
    assert not relative_path.is_absolute(), f"absolute asset path: {value}"
    assert ".." not in relative_path.parts, f"traversal asset path: {value}"
    assert not any(part in {".cache", "cache", "__pycache__"} for part in relative_path.parts), (
        f"cache-backed asset path: {value}"
    )
    resolved = asset_root.joinpath(*relative_path.parts).resolve()
    assert resolved.is_relative_to(asset_root.resolve()), f"escaped asset root: {value}"
    return resolved


def _validate_obj_references(obj_path: Path) -> None:
    material_paths = []
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("mtllib "):
            material_paths.append(_safe_asset_path(obj_path.parent, line.split(maxsplit=1)[1]))
    for material_path in material_paths:
        if not material_path.is_file() and "sources" in obj_path.parts:
            assert material_path.name == "material_0.mtl"
            material_path = material_path.with_name("textured.mtl")
        assert material_path.is_file(), f"missing OBJ material: {material_path}"
        for line in material_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("map_Kd "):
                texture_path = _safe_asset_path(material_path.parent, line.split(maxsplit=1)[1])
                assert texture_path.is_file(), f"missing MTL texture: {texture_path}"


def _validate_manifest(asset_root: Path, manifest: dict[str, object]) -> None:
    assert set(manifest) == {
        "asset_root",
        "collision_policy",
        "conversion",
        "runtime_dependencies",
        "schema_version",
        "source_files",
        "superseded_files",
        "usage",
    }
    assert manifest["schema_version"] == 1
    assert manifest["usage"] == "visual-only"
    collision_policy = manifest["collision_policy"]
    assert isinstance(collision_policy, dict)
    assert collision_policy["authoritative_geometry"] == "existing MuJoCo primitive geoms"
    assert collision_policy["ycb_collision_ply_vendored"] is False
    runtime_dependencies = manifest["runtime_dependencies"]
    assert isinstance(runtime_dependencies, dict)
    assert set(runtime_dependencies.values()) == {False}

    conversion = manifest["conversion"]
    assert isinstance(conversion, dict)
    entries = [*manifest["source_files"], *conversion["generated_files"]]
    declared_paths = set()
    for entry in entries:
        assert isinstance(entry, dict)
        assert entry["usage"] == "visual-only"
        path = _safe_asset_path(asset_root, entry["path"])
        assert entry["path"] not in declared_paths, f"duplicate asset: {entry['path']}"
        declared_paths.add(entry["path"])
        assert path.is_file(), f"missing declared asset: {entry['path']}"
        assert path.stat().st_size == entry["bytes"], f"altered size: {entry['path']}"
        assert _sha256(path) == entry["sha256"], f"altered hash: {entry['path']}"
        if image := entry.get("image"):
            with Image.open(path) as texture:
                assert texture.mode == "RGB", f"non-RGB texture: {entry['path']}"
                assert texture.size == (image["width"], image["height"]), (
                    f"texture dimensions: {entry['path']}"
                )
        if path.suffix == ".obj":
            _validate_obj_references(path)


def _validate_scene_asset_references(scene_path: Path) -> None:
    root = ET.parse(scene_path).getroot()
    for element in root.findall("./asset/mesh") + root.findall("./asset/texture"):
        file_value = element.get("file")
        assert file_value is not None
        path = _safe_asset_path(scene_path.parent, file_value)
        assert path.is_file(), f"missing MJCF asset: {file_value}"
        if path.suffix == ".obj":
            _validate_obj_references(path)

    texture_names = {texture.get("name") for texture in root.findall("./asset/texture")}
    for material in root.findall("./asset/material"):
        texture_name = material.get("texture")
        assert texture_name is None or texture_name in texture_names


def _load_static_model(scene_path: Path) -> mujoco.MjModel:
    policy_modules_before = {name for name in sys.modules if name.startswith("molmo_spaces.policy")}
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    policy_modules_after = {name for name in sys.modules if name.startswith("molmo_spaces.policy")}
    assert policy_modules_after == policy_modules_before
    return model


def _assert_visual_geoms_are_inert(scene_path: Path) -> None:
    root = ET.parse(scene_path).getroot()
    for geom in root.findall(".//geom[@class='visual']"):
        assert geom.get("contype") == "0", f"active visual collision geom: {geom.get('name')}"
        assert geom.get("conaffinity") == "0", f"active visual collision geom: {geom.get('name')}"
        assert geom.get("density") == "0", f"massive visual geom: {geom.get('name')}"


def test_provenance_manifest_is_project_local_and_visual_only() -> None:
    manifest = _manifest()

    assert manifest["schema_version"] == 1
    assert manifest["asset_root"] == "examples/molmoact2_official_sim_eval_yam_box/assets"
    assert manifest["usage"] == "visual-only"
    assert (
        manifest["collision_policy"]["authoritative_geometry"] == "existing MuJoCo primitive geoms"
    )
    assert manifest["collision_policy"]["ycb_collision_ply_vendored"] is False
    assert manifest["conversion"]["status"] == "generated-us-002"
    assert manifest["conversion"]["commands"] == [
        "python3 scripts/convert_molmoact2_yam_visual_assets.py"
    ]
    assert manifest["conversion"]["tool_versions"]["python"]
    assert manifest["conversion"]["tool_versions"]["trimesh"]
    assert manifest["superseded_files"] == [
        {
            "path": "official_yam_tabletop.obj",
            "classification": "generated-legacy-experiment",
            "active_in_current_scene": False,
            "official_maniskill_table_derivative": False,
            "replacement_story": "US-002",
            "usage": "visual-only",
            "note": (
                "Superseded by generated/table meshes converted from the vendored official "
                "ManiSkill table.glb; retained only as an inactive historical artifact."
            ),
        }
    ]
    assert manifest["runtime_dependencies"] == {
        "absolute_paths": False,
        "network_downloads": False,
        "official_checkout": False,
        "service_state": False,
        "user_caches": False,
    }

    source_entries = manifest["source_files"]
    assert {entry["path"] for entry in source_entries} == set(EXPECTED_SOURCES)
    for entry in source_entries:
        relative_path = PurePosixPath(entry["path"])
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        assert entry["usage"] == "visual-only"
        assert entry["immutable"] is True
        assert entry["source_path"]
        assert not PurePosixPath(entry["source_path"]).is_absolute()
        assert entry["upstream_role"]


def test_source_hashes_and_sizes_match_manifest() -> None:
    manifest = _manifest()
    manifest_sources = {entry["path"]: entry for entry in manifest["source_files"]}

    for relative_path, expected in EXPECTED_SOURCES.items():
        path = ASSET_ROOT / relative_path
        assert path.is_file(), relative_path
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]
        assert manifest_sources[relative_path]["bytes"] == expected["bytes"]
        assert manifest_sources[relative_path]["sha256"] == expected["sha256"]


def test_texture_resolutions_are_exact_rgb_64x64() -> None:
    texture_paths = [
        ASSET_ROOT / "sources/mani_skill2_ycb/056_tennis_ball/texture_map.png",
        ASSET_ROOT / "sources/mani_skill2_ycb/073-a_lego_duplo/texture_map.png",
    ]

    for texture_path in texture_paths:
        with Image.open(texture_path) as image:
            assert image.mode == "RGB"
            assert image.size == (64, 64)


def test_provenance_excludes_ycb_collision_meshes() -> None:
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")

    assert not list((ASSET_ROOT / "sources").rglob("collision.ply"))
    assert '"path": "collision.ply"' not in manifest_text


def test_conversion_outputs_match_manifest_hashes_and_image_metadata() -> None:
    manifest = _manifest()
    generated_entries = manifest["conversion"]["generated_files"]

    assert generated_entries
    for entry in generated_entries:
        relative_path = PurePosixPath(entry["path"])
        assert relative_path.parts[0] == "generated"
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        path = ASSET_ROOT / relative_path
        assert path.is_file()
        assert path.stat().st_size == entry["bytes"]
        assert _sha256(path) == entry["sha256"]
        if "image" in entry:
            with Image.open(path) as image:
                assert image.mode == entry["image"]["mode"]
                assert image.size == (entry["image"]["width"], entry["image"]["height"])


def test_conversion_script_and_manifest_document_faithful_transformations() -> None:
    manifest = _manifest()
    transformations = manifest["conversion"]["transformations"]

    assert CONVERSION_SCRIPT.is_file()
    assert transformations["table_geometry"] == (
        "Bake each GLB node transform into its referenced mesh vertices; preserve face order "
        "and per-vertex UV values; write one deterministic OBJ per material assignment."
    )
    assert transformations["table_texture"] == (
        "Decode the embedded TableTop base-color JPEG and deterministically encode RGB PNG "
        "without flipping or otherwise reorienting pixels."
    )
    assert transformations["ycb_geometry"] == (
        "Copy textured.obj bytes while replacing only the unresolved material_0.mtl reference "
        "with textured.mtl."
    )


def test_conversion_preserves_table_vertices_topology_and_uv_coordinates() -> None:
    source_scene = trimesh.load(
        ASSET_ROOT / "sources/maniskill/table/table.glb", force="scene", process=False
    )
    generated_by_node = {
        "tableleg.001": "table_leg_left.obj",
        "tableleg.002": "table_leg_right.obj",
        "tableleg.026": "table_support.obj",
        "tabletop.026": "tabletop.obj",
    }

    for node_name, generated_name in generated_by_node.items():
        transform, geometry_name = source_scene.graph[node_name]
        source_mesh = source_scene.geometry[geometry_name].copy()
        source_mesh.apply_transform(transform)
        generated_mesh = trimesh.load(
            ASSET_ROOT / "generated/table" / generated_name,
            force="mesh",
            process=False,
        )

        assert generated_mesh.vertices == pytest.approx(source_mesh.vertices, abs=5e-10)
        assert np.array_equal(generated_mesh.faces, source_mesh.faces)
        assert generated_mesh.visual.uv == pytest.approx(source_mesh.visual.uv, abs=5e-10)

    source_texture = source_scene.geometry["Cube.002"].visual.material.baseColorTexture
    with Image.open(ASSET_ROOT / "generated/table/tabletop_base_color.png") as generated_texture:
        assert generated_texture.mode == "RGB"
        assert generated_texture.size == source_texture.size
        assert np.array_equal(np.asarray(generated_texture), np.asarray(source_texture))


def test_conversion_preserves_ycb_obj_bytes_except_material_reference() -> None:
    for object_name in ("056_tennis_ball", "073-a_lego_duplo"):
        source = (ASSET_ROOT / f"sources/mani_skill2_ycb/{object_name}/textured.obj").read_bytes()
        generated = (ASSET_ROOT / f"generated/ycb/{object_name}/textured.obj").read_bytes()

        assert generated == source.replace(b"mtllib material_0.mtl", b"mtllib textured.mtl", 1)


def _scene_root() -> ET.Element:
    return ET.parse(SCENE_PATH).getroot()


def _geom_signature(geom: ET.Element) -> dict[str, str | None]:
    physics_attributes = (
        "name",
        "type",
        "size",
        "mesh",
        "pos",
        "quat",
        "contype",
        "conaffinity",
        "group",
        "mass",
        "density",
        "friction",
        "solref",
        "solimp",
        "margin",
        "gap",
        "priority",
    )
    return {attribute: geom.get(attribute) for attribute in physics_attributes}


EXPECTED_COLLISION_SIGNATURES = {
    "floor": {"type": "plane", "size": "2 2 0.1", "pos": "0 0 0"},
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


def test_collision_signature_preserves_all_authoritative_geoms() -> None:
    root = _scene_root()
    parent_by_child = {child: parent for parent in root.iter() for child in parent}
    geoms = {geom.get("name"): geom for geom in root.findall(".//geom")}

    assert set(EXPECTED_COLLISION_SIGNATURES) <= set(geoms)
    for name, expected_attributes in EXPECTED_COLLISION_SIGNATURES.items():
        geom = geoms[name]
        signature = _geom_signature(geom)
        expected_signature = {attribute: None for attribute in signature}
        expected_signature.update(name=name, **expected_attributes)
        assert signature == expected_signature
        expected_body = {
            "floor": "worldbody",
            "obj_073-a_lego_duplo_base": "obj_073-a_lego_duplo",
            "obj_073-a_lego_duplo_stud_1": "obj_073-a_lego_duplo",
            "obj_073-a_lego_duplo_stud_2": "obj_073-a_lego_duplo",
            "obj_073-a_lego_duplo_stud_3": "obj_073-a_lego_duplo",
            "obj_073-a_lego_duplo_stud_4": "obj_073-a_lego_duplo",
            "obj_056_tennis_ball_geom": "obj_056_tennis_ball",
            "open_box_floor": "open_box",
            "open_box_wall_pos_x": "open_box",
            "open_box_wall_neg_x": "open_box",
            "open_box_wall_pos_y": "open_box",
            "open_box_wall_neg_y": "open_box",
        }[name]
        assert parent_by_child[geom].get("name", parent_by_child[geom].tag) == expected_body


def test_visual_mesh_overlays_are_collision_disabled_and_massless() -> None:
    root = _scene_root()
    visual_geoms = [geom for geom in root.findall(".//geom") if geom.get("class") == "visual"]

    assert {geom.get("name") for geom in visual_geoms} == {
        "official_yam_table_leg_left_visual",
        "official_yam_table_leg_right_visual",
        "official_yam_table_support_visual",
        "official_yam_tabletop_visual",
        "obj_073-a_lego_duplo_visual",
        "obj_056_tennis_ball_visual",
    }
    for geom in visual_geoms:
        assert geom.get("type") == "mesh"
        assert geom.get("contype") == "0"
        assert geom.get("conaffinity") == "0"
        assert geom.get("mass") is None
        assert geom.get("density") == "0"


def test_visual_meshes_use_official_textures_and_supersede_legacy_mesh() -> None:
    root = _scene_root()
    assets = root.find("asset")
    assert assets is not None
    mesh_files = {mesh.get("name"): mesh.get("file") for mesh in assets.findall("mesh")}
    texture_files = {
        texture.get("name"): texture.get("file") for texture in assets.findall("texture")
    }
    materials = {material.get("name"): material for material in assets.findall("material")}

    assert "assets/official_yam_tabletop.obj" not in set(mesh_files.values())
    assert texture_files == {
        "official_tabletop_texture": "assets/generated/table/tabletop_base_color.png",
        "tennis_ball_texture": "assets/generated/ycb/056_tennis_ball/texture_map.png",
        "duplo_texture": "assets/generated/ycb/073-a_lego_duplo/texture_map.png",
    }
    assert materials["official_tabletop_mat"].get("texture") == "official_tabletop_texture"
    assert materials["tennis_ball_visual_mat"].get("texture") == "tennis_ball_texture"
    assert materials["duplo_visual_mat"].get("texture") == "duplo_texture"


def test_visual_mesh_alignment_matches_unchanged_collision_primitives() -> None:
    ball = trimesh.load(
        ASSET_ROOT / "generated/ycb/056_tennis_ball/textured.obj",
        force="mesh",
        process=False,
    )
    duplo = trimesh.load(
        ASSET_ROOT / "generated/ycb/073-a_lego_duplo/textured.obj",
        force="mesh",
        process=False,
    )
    tabletop = trimesh.load(
        ASSET_ROOT / "generated/table/tabletop.obj", force="mesh", process=False
    )

    assert np.max(np.abs(ball.bounds.mean(axis=0))) < 1e-4
    assert ball.extents == pytest.approx([0.06653809, 0.0669005, 0.066457], abs=1e-7)
    assert duplo.bounds.mean(axis=0) == pytest.approx([0.0, 2e-8, 1e-8], abs=1e-7)
    assert duplo.extents == pytest.approx([0.064119, 0.032515, 0.023928], abs=1e-7)
    assert tabletop.bounds[1, 2] == pytest.approx(0.525510416, abs=1e-7)

    root = _scene_root()
    bodies = {body.get("name"): body for body in root.findall(".//body")}
    assert bodies["obj_056_tennis_ball"].get("pos") == "-0.300000 -0.220000 0.028000"
    assert bodies["obj_073-a_lego_duplo"].get("pos") == "-0.300000 0.220000 0.025000"
    table_visuals = [
        geom
        for geom in root.findall(".//geom")
        if geom.get("name", "").startswith("official_yam_table")
    ]
    assert {geom.get("pos") for geom in table_visuals} == {"0 0 -0.525510416"}
    assert {geom.get("quat") for geom in table_visuals} == {"0.707106781 0 0 0.707106781"}


def test_asset_manifest_and_all_references_validate_independently() -> None:
    manifest = _manifest()

    _validate_manifest(ASSET_ROOT, manifest)
    _validate_scene_asset_references(SCENE_PATH)
    _assert_visual_geoms_are_inert(SCENE_PATH)


def test_active_scene_loads_all_assets_without_policy_or_rollout() -> None:
    model = _load_static_model(SCENE_PATH)

    assert model.ntex == 3
    assert model.nmat == 8
    assert model.nmesh == 6
    assert model.ngeom == 18
    assert model.nu == 0


def test_asset_validation_detects_changed_hash_in_temporary_copy(tmp_path: Path) -> None:
    copied_assets = tmp_path / "assets"
    shutil.copytree(ASSET_ROOT, copied_assets)
    manifest = _manifest()
    changed_path = copied_assets / manifest["source_files"][0]["path"]
    changed_bytes = bytearray(changed_path.read_bytes())
    changed_bytes[-1] ^= 1
    changed_path.write_bytes(changed_bytes)

    with pytest.raises(AssertionError, match="altered hash"):
        _validate_manifest(copied_assets, manifest)


def test_asset_validation_detects_reduced_texture_resolution(tmp_path: Path) -> None:
    copied_assets = tmp_path / "assets"
    shutil.copytree(ASSET_ROOT, copied_assets)
    manifest = _manifest()
    texture_entry = next(
        entry for entry in manifest["source_files"] if entry["path"].endswith("texture_map.png")
    )
    texture_path = copied_assets / texture_entry["path"]
    with Image.open(texture_path) as texture:
        texture.resize((32, 32)).save(texture_path)
    texture_entry["bytes"] = texture_path.stat().st_size
    texture_entry["sha256"] = _sha256(texture_path)

    with pytest.raises(AssertionError, match="texture dimensions"):
        _validate_manifest(copied_assets, manifest)


@pytest.mark.parametrize(
    ("unsafe_path", "message"),
    [
        ("/tmp/asset.png", "absolute asset path"),
        ("../asset.png", "traversal asset path"),
        (".cache/asset.png", "cache-backed asset path"),
        ("https://example.com/asset.png", "network-backed asset path"),
    ],
)
def test_asset_validation_rejects_unsafe_manifest_paths(unsafe_path: str, message: str) -> None:
    manifest = _manifest()
    manifest["source_files"][0]["path"] = unsafe_path

    with pytest.raises(AssertionError, match=message):
        _validate_manifest(ASSET_ROOT, manifest)


def test_asset_validation_detects_active_visual_collision_geom(tmp_path: Path) -> None:
    copied_scene_root = tmp_path / "scene"
    shutil.copytree(SCENE_PATH.parent, copied_scene_root)
    scene_path = copied_scene_root / "scene.xml"
    root = ET.parse(scene_path).getroot()
    visual_geom = root.find(".//geom[@class='visual']")
    assert visual_geom is not None
    visual_geom.set("contype", "1")
    ET.ElementTree(root).write(scene_path, encoding="unicode")

    with pytest.raises(AssertionError, match="active visual collision geom"):
        _assert_visual_geoms_are_inert(scene_path)


def test_static_model_load_rejects_malformed_mjcf(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.xml"
    scene_path.write_text("<mujoco><worldbody>", encoding="utf-8")

    with pytest.raises(ValueError):
        _load_static_model(scene_path)


def test_static_model_load_rejects_missing_texture(tmp_path: Path) -> None:
    copied_scene_root = tmp_path / "scene"
    shutil.copytree(SCENE_PATH.parent, copied_scene_root)
    scene_path = copied_scene_root / "scene.xml"
    missing_texture = copied_scene_root / "assets/generated/table/tabletop_base_color.png"
    missing_texture.unlink()

    with pytest.raises(ValueError, match="tabletop_base_color.png"):
        _load_static_model(scene_path)
