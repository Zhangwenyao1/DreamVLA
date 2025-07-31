import os
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

dataset_path = '/inspire/hdd/global_user/guchun-240107140023/task_ABC_D'
traj_path = '/inspire/hdd/project/robotsimulation/guchun-240107140023/Wenyao/DreamVLA/co-tracker/calvin_dense_k10/task_ABC_D'
output_path = '/inspire/ssd/project/robotsimulation/guchun-240107140023/Wenyao/calvin_datasets_mergedall/task_ABC_D'

splits = ['training']

def _convert(file):
    # try:
    track_label_static = dict(np.load(os.path.join(traj_path, 'rgb_static', split, file)))
    track_label_gripper = dict(np.load(os.path.join(traj_path, 'rgb_gripper', split, file)))

    track_keys = ['tracks', 'visibility']

    episode = dict(np.load(os.path.join(dataset_path, split, f'episode_{int(file.split(".")[0]):07d}' + '.npz')))
    episode['traj_static'] = track_label_static['tracks']
    episode['traj_gripper'] = track_label_gripper['tracks']
    episode['visibility_static'] = track_label_static['visibility']
    episode['visibility_gripper'] = track_label_gripper['visibility']
    output_file = os.path.join(output_path, split, f'episode_{int(file.split(".")[0]):07d}.npz')
    
    np.savez_compressed(output_file, **episode)
    # except Exception as e:
    #     print(f"Error processing {file}: {e}")

for split in splits:
    all_files = os.listdir(os.path.join(traj_path, 'rgb_static', split))
    os.makedirs(os.path.join(output_path, split), exist_ok=True)
    
    # for file in all_files:
    #     _convert(file)
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(_convert, file) for file in all_files]

    for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        pass