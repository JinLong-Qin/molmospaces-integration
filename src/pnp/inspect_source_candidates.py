from __future__ import annotations
import io, json, re, tarfile, tempfile, os
from pathlib import Path
import h5py, numpy as np, zstandard as zstd
WORK=Path(os.environ.get('MOLMOSPACES_PNP_WORKDIR', 'runtime/mimicgen_pick_and_place'))
TAR=WORK/'data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards/00000.tar'
houses=[1670,1716,2423,3080,5790,9695,4519,1536,9042,9270]
with tarfile.open(TAR,'r') as outer:
    for house in houses:
        print('### HOUSE', house)
        member=outer.getmember(f'FrankaPickAndPlaceOmniCamConfig_house_{house}.tar.zst')
        comp=outer.extractfile(member).read()
        reader=zstd.ZstdDecompressor().stream_reader(io.BytesIO(comp))
        try:
            with tarfile.open(fileobj=reader, mode='r|') as inner:
                for hm in inner:
                    if not hm.isfile(): continue
                    if hm.name.endswith('.mp4') and 'randomized_zed2_analogue_1' in hm.name:
                        print('MP4', hm.name, hm.size)
                    if hm.name.endswith('.h5'):
                        data=inner.extractfile(hm).read()
                        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                            f.write(data); tmp=f.name
                        try:
                            with h5py.File(tmp,'r') as h:
                                print('H5', hm.name, 'keys', list(h.keys()), 'valid_mask', np.asarray(h.get('valid_traj_mask',[])).tolist())
                                keys=sorted([k for k in h.keys() if k.startswith('traj_')], key=lambda s:int(s.split('_')[1]))
                                for k in keys:
                                    g=h[k]
                                    success=np.asarray(g.get('success',[]), dtype=bool)
                                    obs={}
                                    try: obs=json.loads(bytes(g['obs_scene'][()]).rstrip(b'\0'))
                                    except Exception as e: obs={'ERR':repr(e)}
                                    first=int(np.argmax(success)) if len(success) and success.any() else -1
                                    pers=first>=0 and bool(success[-1]) and bool(success[first:].all())
                                    print(' TRAJ', k, 'len', len(success), 'final', bool(success[-1]) if len(success) else None, 'first', first, 'pers', pers, 'obj', obs.get('object_name'), 'place', obs.get('place_receptacle_name'), 'task', obs.get('task_description'))
                        finally:
                            os.unlink(tmp)
        finally:
            try: reader.close()
            except Exception: pass
