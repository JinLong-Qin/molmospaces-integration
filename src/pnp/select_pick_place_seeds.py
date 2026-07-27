from __future__ import annotations
import io, json, os, re, tarfile, tempfile
from pathlib import Path
import h5py, numpy as np, zstandard as zstd

WORK = Path(os.environ.get("MOLMOSPACES_PNP_WORKDIR", "runtime/mimicgen_pick_and_place"))
TAR = WORK / "data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar"
OUT = WORK / "artifacts/seeds"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
manifest = {"dataset": "FrankaPickAndPlaceOmniCamConfig", "split": "val_shards/00000.tar", "seeds": []}

def scan_h5_bytes(data: bytes):
    candidates=[]
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        f.write(data)
        tmp=f.name
    try:
        with h5py.File(tmp,"r") as h:
            mask=np.asarray(h.get("valid_traj_mask", []), dtype=bool)
            keys=sorted([x for x in h.keys() if x.startswith("traj_")], key=lambda s:int(s.split("_")[1]))
            for k in keys:
                ti=int(k.split("_")[1])
                if len(mask) and not bool(mask[ti]):
                    continue
                g=h[k]
                if "success" not in g:
                    continue
                success=np.asarray(g["success"][:], dtype=bool)
                reward=float(np.asarray(g["rewards"][:]).sum()) if "rewards" in g else 0.0
                if not (len(success) and bool(success[-1])):
                    continue
                obs_scene={}
                if "obs_scene" in g:
                    try:
                        obs_scene=json.loads(bytes(g["obs_scene"][()]).rstrip(b"\0"))
                    except Exception:
                        obs_scene={}
                task_info={}
                try:
                    row=bytes(g["obs/extra/task_info"][-1]).rstrip(b"\0")
                    task_info=json.loads(row.decode("utf-8")) if row else {}
                except Exception:
                    pass
                candidates.append({
                    "traj_index": ti,
                    "traj_key": k,
                    "length": int(len(success)),
                    "reward_sum": reward,
                    "success_final": bool(success[-1]),
                    "object_name": obs_scene.get("object_name"),
                    "pickup_obj_name": obs_scene.get("pickup_obj_name"),
                    "place_receptacle_name": obs_scene.get("place_receptacle_name"),
                    "task_type": obs_scene.get("task_type"),
                    "task_description": obs_scene.get("task_description"),
                    "final_task_info": task_info,
                    "obs_scene_keys": sorted(list(obs_scene.keys()))[:40],
                })
    finally:
        os.unlink(tmp)
    return candidates

selected=None
with tarfile.open(TAR, "r") as outer:
    members=[m for m in outer.getmembers() if m.isfile() and m.name.endswith(".tar.zst") and "house_" in m.name]
    for m in members:
        house_match=re.search(r"house_(\d+)", m.name)
        house_id=int(house_match.group(1)) if house_match else -1
        print("HOUSE", house_id, m.name, flush=True)
        comp=outer.extractfile(m).read()
        reader=zstd.ZstdDecompressor().stream_reader(io.BytesIO(comp))
        with tarfile.open(fileobj=reader, mode="r|") as inner:
            for hm in inner:
                if not hm.isfile() or not hm.name.endswith(".h5"):
                    continue
                data=inner.extractfile(hm).read()
                cands=scan_h5_bytes(data)
                if cands:
                    batch_match=re.search(r"trajectories_batch_(\d+)_", hm.name)
                    batch_id=int(batch_match.group(1)) if batch_match else -1
                    seed_meta=dict(cands[0])
                    seed_meta.update({
                        "seed_index": 0,
                        "house_id":house_id,
                        "batch_id":batch_id,
                        "source_outer_member":m.name,
                        "source_inner_h5":hm.name,
                        "raw_h5":f"seed_00__house{house_id}__batch{batch_id}.h5",
                    })
                    out_h5=RAW/seed_meta["raw_h5"]
                    out_h5.write_bytes(data)
                    selected=seed_meta
                    print("SELECTED", json.dumps(selected, indent=2, ensure_ascii=False), flush=True)
                    break
        try:
            reader.close()
        except Exception:
            pass
        if selected:
            break
if not selected:
    raise SystemExit("no successful PnP trajectory found")
manifest["seeds"].append(selected)
(OUT/"pnp_seed_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
