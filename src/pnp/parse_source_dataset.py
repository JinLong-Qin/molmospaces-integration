from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

MIM = Path(os.environ.get('MIMICGEN_ROOT', '../mimicgen'))
sys.path.insert(0, str(MIM/'vendor/mimicgen'))
sys.path.insert(0, str(MIM/'vendor/robomimic'))
from mimicgen.utils.file_utils import parse_source_dataset

SRC = Path(os.environ.get('PNP_SOURCE_HDF5', 'work/current/mimicgen_pick_and_place/artifacts/seeds/robomimic_pnp_seed00_aligned.hdf5'))
OUT = SRC.parent / 'robomimic_pnp_seed00_parse_result.json'
# Six detected subtasks + final residual segment. Offsets zero for first gate to avoid hiding boundary problems.
signals = ['pregrasp_done', 'grasp_done', 'gripper_closed', 'lift_done', 'preplace_done', 'place_success', None]
offsets = [(0,0)] * len(signals)
infos, indices, names, ranges = parse_source_dataset(
    dataset_path=str(SRC),
    demo_keys=['demo_0'],
    subtask_term_signals=signals,
    subtask_term_offset_ranges=offsets,
)
res = {
    'source': str(SRC),
    'demo_keys': ['demo_0'],
    'subtask_term_signals': [str(x) if x is not None else None for x in names],
    'subtask_term_offset_ranges': ranges,
    'subtask_indices_shape': list(indices.shape),
    'subtask_indices': indices.tolist(),
    'segment_lengths': [[int(b-a) for a,b in demo] for demo in indices.tolist()],
    'datagen_info_count': len(infos),
    'datagen_shapes': {
        'eef_pose': list(infos[0].eef_pose.shape),
        'target_pose': list(infos[0].target_pose.shape),
        'gripper_action': list(infos[0].gripper_action.shape),
        'object_poses': {k:list(v.shape) for k,v in infos[0].object_poses.items()},
        'subtask_term_signals': {k:list(v.shape) for k,v in infos[0].subtask_term_signals.items()},
    },
}
OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False))
