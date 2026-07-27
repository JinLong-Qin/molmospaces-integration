from __future__ import annotations
import argparse, json, os
from pathlib import Path
import h5py
import numpy as np

WORK = Path(os.environ.get('MOLMOSPACES_PNP_WORKDIR', 'work/current/mimicgen_pick_and_place'))

def decode_json_row(row) -> dict:
    raw = bytes(row).rstrip(b'\0')
    return json.loads(raw or b'{}')

def pose_to_homogeneous(pose7: np.ndarray) -> np.ndarray:
    xyz = pose7[:3]
    w, x, y, z = pose7[3], pose7[4], pose7[5], pose7[6]
    norm = float(np.sqrt(w*w+x*x+y*y+z*z))
    if norm < 1e-9:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    else:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ], dtype=np.float32)
    H = np.eye(4, dtype=np.float32)
    H[:3,:3] = R
    H[:3,3] = xyz.astype(np.float32)
    return H

def batched_pose_to_homogeneous(pose_batch: np.ndarray) -> np.ndarray:
    return np.stack([pose_to_homogeneous(p) for p in pose_batch], axis=0)

def gripper_qpos_from_rows(rows, n):
    out = np.zeros((n,1), dtype=np.float32)
    last = 0.0
    for i in range(n):
        d = decode_json_row(rows[i])
        if 'gripper' in d and len(d['gripper']) >= 1:
            last = float(d['gripper'][0])
        out[i,0] = last
    return out

def cumulative_signal_from_index(n: int, idx: int) -> np.ndarray:
    sig = np.zeros(n, dtype=np.uint8)
    if 0 <= idx < n:
        sig[idx:] = 1
    return sig

def first_index_where(arr, pred):
    for i, v in enumerate(arr):
        if pred(v):
            return i
    return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-index', type=int, default=0)
    ap.add_argument('--out', type=Path, default=WORK/'artifacts/seeds/robomimic_pnp_seed00_aligned.hdf5')
    args = ap.parse_args()
    manifest = json.loads((WORK/'artifacts/seeds/pnp_seed_manifest.json').read_text())
    seed = manifest['seeds'][args.seed_index]
    raw_h5 = WORK/'artifacts/seeds/raw'/seed['raw_h5']
    traj = seed['traj_key']
    replay_dir = WORK/f'artifacts/replay_pnp_exact/seed_{args.seed_index:02d}'
    pickup_rel_obs = np.load(replay_dir/'pickup_obj_pose_rel_observations.npy')
    place_rel_obs = np.load(replay_dir/'place_receptacle_pose_rel_observations.npy')
    target_pose_all = np.load(replay_dir/'target_pose_rel_actions.npy')

    with h5py.File(raw_h5, 'r') as f:
        g = f[traj]
        obs_scene = json.loads(bytes(g['obs_scene'][()]).rstrip(b'\0'))
        commanded_all = [decode_json_row(x) for x in g['actions/commanded_action'][:]]
        real_action_rows = [i for i,d in enumerate(commanded_all) if 'arm' in d and 'gripper' in d and len(d.get('arm', [])) == 7 and len(d.get('gripper', [])) >= 1]
        if real_action_rows[0] != 1:
            raise RuntimeError(f'unexpected first real action row {real_action_rows[0]}')
        # Use only real control rows. Last row in this seed is an episode success marker, not arm/gripper control.
        n = len(real_action_rows)
        assert n == 153, n
        actions = np.zeros((n,8), dtype=np.float32)
        for j, row_i in enumerate(real_action_rows):
            d = commanded_all[row_i]
            actions[j,:7] = np.asarray(d['arm'], dtype=np.float32)
            actions[j,7] = float(d['gripper'][0])
        pre_obs = np.asarray(real_action_rows, dtype=int) - 1
        post_obs = np.asarray(real_action_rows, dtype=int)
        tcp7 = g['obs/extra/tcp_pose'][pre_obs].astype(np.float32)
        eef_hmat = batched_pose_to_homogeneous(tcp7)
        states = g['env_states/articulations/panda'][pre_obs].astype(np.float32)
        rewards = g['rewards'][post_obs].astype(np.float32)
        success = g['success'][post_obs].astype(bool)
        phases_pre = g['obs/extra/policy_phase'][pre_obs].astype(np.int64)
        phases_post = g['obs/extra/policy_phase'][post_obs].astype(np.int64)
        gripper_qpos = gripper_qpos_from_rows(g['obs/agent/qpos'], len(g['obs/agent/qpos']))[pre_obs]
        pickup_obj_name = obs_scene.get('object_name')
        place_receptacle_name = obs_scene.get('place_receptacle_name')

    # Align datagen arrays to real control rows / pre-step observations.
    pickup_obj_pose = pickup_rel_obs[pre_obs].astype(np.float32)
    place_receptacle_pose = place_rel_obs[pre_obs].astype(np.float32)
    target_pose = target_pose_all[:n].astype(np.float32)
    if target_pose.shape != (n,4,4):
        raise RuntimeError(f'target_pose shape {target_pose.shape}, n={n}')

    # PnP phase-derived subtask boundaries. These are cumulative 0->1 signals for official MimicGen parser.
    idx_pregrasp_done = first_index_where(phases_post, lambda p: int(p) >= 3)
    idx_grasp_done = first_index_where(phases_post, lambda p: int(p) >= 4)
    idx_gripper_closed = first_index_where(phases_post, lambda p: int(p) >= 5)
    idx_lift_done = first_index_where(phases_post, lambda p: int(p) >= 6)
    idx_preplace_done = first_index_where(phases_post, lambda p: int(p) >= 7)
    idx_place_success = int(np.argmax(success)) if success.any() else -1
    signals = {
        'pregrasp_done': cumulative_signal_from_index(n, idx_pregrasp_done),
        'grasp_done': cumulative_signal_from_index(n, idx_grasp_done),
        'gripper_closed': cumulative_signal_from_index(n, idx_gripper_closed),
        'lift_done': cumulative_signal_from_index(n, idx_lift_done),
        'preplace_done': cumulative_signal_from_index(n, idx_preplace_done),
        'place_success': cumulative_signal_from_index(n, idx_place_success),
    }
    signal_indices = {k: int(np.argmax(v)) if v.any() else -1 for k,v in signals.items()}
    if any(v < 0 for v in signal_indices.values()):
        raise RuntimeError(f'missing subtask signal {signal_indices}')
    # Ensure strictly increasing enough for zero offset parser sanity.
    ordered = [signal_indices[k] for k in ['pregrasp_done','grasp_done','gripper_closed','lift_done','preplace_done','place_success']]
    if ordered != sorted(ordered):
        raise RuntimeError(f'non-monotonic signal indices {signal_indices}')

    dones = np.zeros(n, dtype=np.int32)
    if success.any():
        dones[int(np.argmax(success)):] = 1
    else:
        dones[-1] = 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    with h5py.File(args.out, 'w') as out:
        data = out.create_group('data')
        data.attrs['env_args'] = json.dumps({
            'env_name': 'MolmoSpacesPickAndPlaceEnv',
            'env_kwargs': {'config': 'FrankaPickAndPlaceOmniCamConfig', 'house_id': int(seed['house_id'])},
            'type': 2,
            'note': 'MolmoSpaces adapter TBD; source parsed for MimicGen PnP integration',
        })
        data.attrs['total'] = n
        demo = data.create_group('demo_0')
        demo.create_dataset('actions', data=actions, compression='gzip')
        demo.create_dataset('states', data=states, compression='gzip')
        demo.create_dataset('dones', data=dones)
        demo.create_dataset('rewards', data=rewards)
        obs = demo.create_group('obs')
        obs.create_dataset('robot0_eef_pos', data=tcp7[:,:3])
        obs.create_dataset('robot0_eef_quat', data=tcp7[:,3:7])
        obs.create_dataset('robot0_gripper_qpos', data=gripper_qpos)
        obs.create_dataset('object_pos', data=pickup_obj_pose[:,:3,3])
        obs.create_dataset('place_receptacle_pos', data=place_receptacle_pose[:,:3,3])
        obs.create_dataset('tcp_pos', data=tcp7[:,:3])
        obs.create_dataset('tcp_quat', data=tcp7[:,3:7])
        obs.create_dataset('policy_phase', data=phases_pre)
        dgi = demo.create_group('datagen_info')
        dgi.attrs['env_interface_name'] = 'MG_MolmoSpacesPickAndPlace'
        dgi.attrs['env_interface_type'] = 'molmospaces'
        dgi.create_dataset('eef_pose', data=eef_hmat, compression='gzip')
        dgi.create_dataset('target_pose', data=target_pose, compression='gzip')
        dgi.create_dataset('gripper_action', data=actions[:,7:8].astype(np.float32))
        op = dgi.create_group('object_poses')
        op.create_dataset('pickup_obj', data=pickup_obj_pose, compression='gzip')
        op.create_dataset('place_receptacle', data=place_receptacle_pose, compression='gzip')
        sig = dgi.create_group('subtask_term_signals')
        for k,v in signals.items():
            sig.create_dataset(k, data=v)
        demo.attrs['house_id'] = int(seed['house_id'])
        demo.attrs['batch_id'] = int(seed['batch_id'])
        demo.attrs['traj_index'] = int(seed['traj_index'])
        demo.attrs['source_h5'] = seed['raw_h5']
        demo.attrs['num_samples'] = n
        demo.attrs['source_observations'] = int(seed['length'])
        demo.attrs['alignment'] = 'real commanded_action rows only: rows 1..153; final success marker row excluded; pre_obs=row-1, post_obs=row'
        demo.attrs['seed_kind'] = 'synthetic_planner_expert'
        demo.attrs['task_type'] = 'pick_and_place'
        demo.attrs['task_description'] = obs_scene.get('task_description', '')
        demo.attrs['pickup_obj_name'] = pickup_obj_name
        demo.attrs['place_receptacle_name'] = place_receptacle_name
        demo.attrs['subtask_signal_indices'] = json.dumps(signal_indices)
        out.attrs['provenance'] = 'MolmoBot-Data allenai/molmobot-data / FrankaPickAndPlaceOmniCamConfig'
        out.attrs['seed_kind'] = 'synthetic_planner_expert (cuRobo/scripted). NOT human demonstration.'
        out.attrs['n_demos'] = 1
    print('[done]', args.out)
    print('n', n, 'success_first', int(np.argmax(success)) if success.any() else -1, 'success_final', bool(success[-1]))
    print('signal_indices', json.dumps(signal_indices, indent=2))

if __name__ == '__main__':
    main()
