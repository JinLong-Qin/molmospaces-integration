from __future__ import annotations
import io, json, os, re, tarfile, tempfile
from pathlib import Path
import h5py, numpy as np, zstandard as zstd

WORK = Path(os.environ.get("MOLMOSPACES_PNP_WORKDIR", "runtime/mimicgen_pick_and_place"))
TAR = WORK / "data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar"
OUT = WORK / "artifacts/seeds"
RAW = OUT / "raw_homogeneous_potato_tomato_bowl"
RAW.mkdir(parents=True, exist_ok=True)
# 2423/traj1 is visually sent but valid_traj_mask=False, so exclude from official source set.
wanted = [
    (1670, 0, "white potato -> rustic clay bowl"),
    (1716, 1, "irish potato -> round gray bowl"),
    (3080, 0, "small red potato -> rustic shallow bowl"),
    (3080, 1, "small red potato -> rustic round brown ceramic bowl"),
    (5790, 0, "irish potato -> rustic shallow bowl"),
    (5790, 1, "brown potato -> rustic shallow bowl"),
    (9695, 0, "red potato -> rustic dark ceramic bowl A"),
    (9695, 1, "red potato -> rustic dark ceramic bowl B"),
    (4519, 0, "red tomato -> blue/white ceramic bowl"),
    (
        1536,
        1,
        "replacement: tomato -> terracotta bowl; 2423/traj1 excluded because valid_traj_mask=False",
    ),
]
by_house = {}
for h, t, d in wanted:
    by_house.setdefault(h, []).append((t, d))


def scan(data, house_id, batch_id, outer_name, inner_h5):
    rows = []
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        with h5py.File(tmp, "r") as h:
            mask = np.asarray(h.get("valid_traj_mask", []), dtype=bool)
            keys = sorted(
                [k for k in h.keys() if k.startswith("traj_")], key=lambda s: int(s.split("_")[1])
            )
            for k in keys:
                ti = int(k.split("_")[1])
                valid_mask = bool(mask[ti]) if ti < len(mask) else True
                g = h[k]
                success = np.asarray(g.get("success", []), dtype=bool)
                obs = {}
                try:
                    obs = json.loads(bytes(g["obs_scene"][()]).rstrip(b"\0"))
                except Exception:
                    pass
                first = int(np.argmax(success)) if len(success) and success.any() else -1
                pers = first >= 0 and bool(success[-1]) and bool(success[first:].all())
                rows.append(
                    {
                        "traj_index": ti,
                        "traj_key": k,
                        "valid_traj_mask": valid_mask,
                        "length": int(len(success)),
                        "success_final": bool(success[-1]) if len(success) else False,
                        "first_success": first,
                        "success_persistent_to_end_metadata": pers,
                        "object_name": obs.get("object_name"),
                        "pickup_obj_name": obs.get("pickup_obj_name"),
                        "place_receptacle_name": obs.get("place_receptacle_name"),
                        "task_type": obs.get("task_type"),
                        "task_description": obs.get("task_description"),
                        "house_id": house_id,
                        "batch_id": batch_id,
                        "source_outer_member": outer_name,
                        "source_inner_h5": inner_h5,
                    }
                )
    finally:
        os.unlink(tmp)
    return rows


found = []
excluded = []
with tarfile.open(TAR, "r") as outer:
    for house_id, wants in by_house.items():
        member_name = f"FrankaPickAndPlaceOmniCamConfig_house_{house_id}.tar.zst"
        comp = outer.extractfile(outer.getmember(member_name)).read()
        reader = zstd.ZstdDecompressor().stream_reader(io.BytesIO(comp))
        try:
            h5_bytes = None
            inner_h5 = None
            batch_id = -1
            with tarfile.open(fileobj=reader, mode="r|") as inner:
                for hm in inner:
                    if hm.isfile() and hm.name.endswith(".h5"):
                        h5_bytes = inner.extractfile(hm).read()
                        inner_h5 = hm.name
                        m = re.search(r"trajectories_batch_(\d+)_", hm.name)
                        batch_id = int(m.group(1)) if m else -1
                        break
            rows = scan(h5_bytes, house_id, batch_id, member_name, inner_h5)
            by_traj = {r["traj_index"]: r for r in rows}
            for traj_index, note in wants:
                r = dict(by_traj[traj_index])
                if not (
                    r["valid_traj_mask"]
                    and r["success_final"]
                    and r["success_persistent_to_end_metadata"]
                ):
                    raise RuntimeError(
                        "candidate failed official pregate " + json.dumps(r, ensure_ascii=False)
                    )
                raw_name = f"hom_seed_{len(found):02d}__house{house_id}__batch{batch_id}.h5"
                (RAW / raw_name).write_bytes(h5_bytes)
                r.update(
                    {
                        "seed_index": len(found),
                        "candidate_note": note,
                        "raw_h5": raw_name,
                        "raw_h5_dir": str(RAW.relative_to(OUT)),
                        "seed_kind": "synthetic_planner_expert",
                        "selection_family": "potato_tomato_to_bowl",
                    }
                )
                found.append(r)
        finally:
            try:
                reader.close()
            except Exception:
                pass
manifest = {
    "dataset": "FrankaPickAndPlaceOmniCamConfig",
    "split": "val_shards/00000.tar",
    "selection_note": "homogeneous potato/tomato -> bowl source set; 2423/traj1 visual candidate excluded due valid_traj_mask=False and replaced by 1536/traj1",
    "seed_kind": "synthetic_planner_expert (cuRobo/scripted), NOT human demonstrations",
    "excluded_visual_candidates": [
        {"house_id": 2423, "traj_index": 1, "reason": "valid_traj_mask=False"}
    ],
    "seeds": found,
}
out = OUT / "pnp_seed_manifest_homogeneous_potato_tomato_bowl.json"
out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(
    json.dumps(
        {
            "out": str(out),
            "n": len(found),
            "seeds": [
                {
                    k: s[k]
                    for k in [
                        "seed_index",
                        "house_id",
                        "traj_key",
                        "valid_traj_mask",
                        "task_description",
                        "first_success",
                        "length",
                        "raw_h5",
                    ]
                }
                for s in found
            ],
        },
        indent=2,
        ensure_ascii=False,
    )
)
