import os
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

dataset_path = 'calvin_datasets/task_ABC_D'
dino_path = '/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D'
sam_path = '/mnt/afs/wyzhang/code/segment-anything/calvin_sam/task_ABC_D'
output_path = 'calvin_datasets_with_dinosam/task_ABC_D'

splits = ['validation', 'training']

def _convert(file):
    try:
        dino_static = torch.load(os.path.join(dino_path, 'rgb_static', split, file)).to(torch.float32).numpy()
        dino_gripper = torch.load(os.path.join(dino_path, 'rgb_gripper', split, file)).to(torch.float32).numpy()
        sam_static = torch.load(os.path.join(sam_path, 'rgb_static', split, file)).to(torch.float32).numpy()
        sam_gripper = torch.load(os.path.join(sam_path, 'rgb_gripper', split, file)).to(torch.float32).numpy()

        episode = dict(np.load(os.path.join(dataset_path, split, f'episode_{int(file.split(".")[0]):07d}' + '.npz')))
        episode['dino_static'] = dino_static
        episode['dino_gripper'] = dino_gripper
        episode['sam_static'] = sam_static
        episode['sam_gripper'] = sam_gripper
        output_file = os.path.join(output_path, split, f'episode_{int(file.split(".")[0]):07d}.npz')
        
        np.savez_compressed(output_file, **episode)
    except Exception as e:
        print(f"Error processing {file}: {e}")

for split in splits:
    all_files = os.listdir(os.path.join(sam_path, 'rgb_static', split))
    os.makedirs(os.path.join(output_path, split), exist_ok=True)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_convert, file) for file in all_files]

    for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        pass