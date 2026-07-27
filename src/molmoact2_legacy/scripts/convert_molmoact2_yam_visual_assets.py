#!/usr/bin/env python3
"""Generate project-local MuJoCo visual assets from vendored official sources."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import struct
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "examples/molmoact2_official_sim_eval_yam_box/assets"
SOURCE_ROOT = ASSET_ROOT / "sources"
GENERATED_ROOT = ASSET_ROOT / "generated"
MANIFEST_PATH = ASSET_ROOT / "visual_asset_provenance.json"

TABLE_NODES = {
    "tableleg.001": "table/table_leg_left.obj",
    "tableleg.002": "table/table_leg_right.obj",
    "tableleg.026": "table/table_support.obj",
    "tabletop.026": "table/tabletop.obj",
}
YCB_OBJECTS = ("056_tennis_ball", "073-a_lego_duplo")


def _format_float(value: float) -> str:
    rounded = 0.0 if abs(value) < 5e-11 else value
    return f"{rounded:.9f}"


def _write_obj(path: Path, mesh: trimesh.Trimesh, material_name: str) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    uv = np.asarray(mesh.visual.uv)
    normals = np.asarray(mesh.vertex_normals)
    if len(uv) != len(vertices) or len(normals) != len(vertices):
        raise ValueError(f"Expected one UV and normal per vertex for {path.name}")

    lines = [f"usemtl {material_name}"]
    lines.extend("v " + " ".join(_format_float(value) for value in vertex) for vertex in vertices)
    lines.extend("vt " + " ".join(_format_float(value) for value in texcoord) for texcoord in uv)
    lines.extend("vn " + " ".join(_format_float(value) for value in normal) for normal in normals)
    for face in faces:
        indices = [int(index) + 1 for index in face]
        lines.append("f " + " ".join(f"{index}/{index}/{index}" for index in indices))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def _extract_glb_image(glb_path: Path, image_name: str) -> bytes:
    data = glb_path.read_bytes()
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise ValueError(f"Invalid GLB header: {glb_path}")

    offset = 12
    chunks: dict[bytes, bytes] = {}
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunks[chunk_type] = data[offset : offset + chunk_length]
        offset += chunk_length
    document = json.loads(chunks[b"JSON"].rstrip(b" \0"))
    binary = chunks[b"BIN\0"]
    image = next(item for item in document["images"] if item["name"] == image_name)
    view = document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    return binary[start : start + view["byteLength"]]


def _convert_table() -> None:
    glb_path = SOURCE_ROOT / "maniskill/table/table.glb"
    scene = trimesh.load(glb_path, force="scene", process=False)
    for node_name, relative_output in TABLE_NODES.items():
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        _write_obj(
            GENERATED_ROOT / relative_output,
            mesh,
            mesh.visual.material.name,
        )

    image_bytes = _extract_glb_image(glb_path, "Wood052_1K_Color")
    texture_path = GENERATED_ROOT / "table/tabletop_base_color.png"
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as image:
        image.convert("RGB").save(texture_path, format="PNG", compress_level=9, optimize=False)


def _convert_ycb() -> None:
    for object_name in YCB_OBJECTS:
        source_dir = SOURCE_ROOT / "mani_skill2_ycb" / object_name
        output_dir = GENERATED_ROOT / "ycb" / object_name
        output_dir.mkdir(parents=True, exist_ok=True)
        obj_text = (source_dir / "textured.obj").read_text(encoding="ascii")
        if obj_text.count("mtllib material_0.mtl") != 1:
            raise ValueError(f"Unexpected material reference in {object_name}/textured.obj")
        converted = obj_text.replace("mtllib material_0.mtl", "mtllib textured.mtl", 1)
        (output_dir / "textured.obj").write_text(converted, encoding="ascii", newline="\n")
        shutil.copyfile(source_dir / "textured.mtl", output_dir / "textured.mtl")
        shutil.copyfile(source_dir / "texture_map.png", output_dir / "texture_map.png")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generated_entry(path: Path) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": path.relative_to(ASSET_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "usage": "visual-only",
    }
    if path.suffix.lower() == ".png":
        with Image.open(path) as image:
            entry["image"] = {"mode": image.mode, "width": image.width, "height": image.height}
    return entry


def _update_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    generated_files = sorted(
        path for path in GENERATED_ROOT.rglob("*") if path.is_file() and path.name != ".gitkeep"
    )
    manifest["conversion"] = {
        "status": "generated-us-002",
        "commands": ["python3 scripts/convert_molmoact2_yam_visual_assets.py"],
        "tool_versions": {
            "python": platform.python_version(),
            "trimesh": trimesh.__version__,
            "pillow": Image.__version__,
            "numpy": np.__version__,
        },
        "generated_root": "generated",
        "transformations": {
            "table_geometry": (
                "Bake each GLB node transform into its referenced mesh vertices; preserve face "
                "order and per-vertex UV values; write one deterministic OBJ per material assignment."
            ),
            "table_texture": (
                "Decode the embedded TableTop base-color JPEG and deterministically encode RGB PNG "
                "without flipping or otherwise reorienting pixels."
            ),
            "ycb_geometry": (
                "Copy textured.obj bytes while replacing only the unresolved material_0.mtl "
                "reference with textured.mtl."
            ),
            "ycb_materials_and_textures": (
                "Copy the vendored textured.mtl and 64x64 RGB texture_map.png bytes unchanged."
            ),
        },
        "generated_files": [_generated_entry(path) for path in generated_files],
        "note": (
            "Generated files are visual-only. The active MJCF retains all primitive collision "
            "geoms as authoritative and adds collision-disabled, zero-density render overlays."
        ),
    }
    manifest["superseded_files"] = [
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
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    _convert_table()
    _convert_ycb()
    _update_manifest()


if __name__ == "__main__":
    main()
