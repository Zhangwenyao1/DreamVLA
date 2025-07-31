import os
import re
import random
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import h5py
import tqdm
import argparse
from pathlib import Path
import yaml
from PIL import Image
import time
import json
import math
# from sentence_transformers import SentenceTransformer

### Rotation ###
from scipy.spatial.transform import Rotation


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def extract_task_information(file_name, path):
    """
    Extracts task information from the given file name.
    """
    global dataset_name
    # Regular expression pattern to extract the task name
    if dataset_name == 'libero_10':
        pattern = r'{}/((.+)_SCENE[0-9]+_(.+))_demo\.hdf5'.format(path)
    else:
        pattern = r'{}/(.*)_demo\.hdf5'.format(re.escape(path))

    # Extracting the task name
    match = re.search(pattern, file_name)
    if dataset_name == 'libero_10':
        print(match.group(3).lower().replace("_", " "))
        return match.group(1).lower() if match else None, match.group(3).lower().replace("_", " ")
    else:
        task_full = match.group(1).lower()
        task_simple = task_full.replace("_", " ")
        print(task_simple)
        return task_full, task_simple

class DatasetConverter:
    def __init__(
        self,
        src_dir: str,
        tgt_dir: str,
        rank: int,
        num_worker: int,
        start_episode_idx,
        end_episode_idx,
    ):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.rank = rank
        self.num_worker = num_worker
        self.start_episode_idx = start_episode_idx
        self.end_episode_idx = end_episode_idx
        # self.lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def process_episode(self, episode_dir, language_instructions, demo_data, episode_index, episode_index_in_task):
        i = episode_index_in_task
        # get episode dir
        episode_dir.mkdir(exist_ok=True)

        ### Get agent's view camera
        obs = np.array(demo_data['demo_{}'.format(i)]['obs']['agentview_rgb'])
        # obs = obs.transpose(0,3,1,2)
        
        ### Get wrist's view camera
        obs_wrist = np.array(demo_data['demo_{}'.format(i)]['obs']['eye_in_hand_rgb'])
        # obs_wrist = obs_wrist.transpose(0,3,1,2)

        ### Get actions
        action = np.array(demo_data['demo_{}'.format(i)]['actions'])  # -1 open, 1 close
        
        joint_state = np.array(demo_data['demo_{}'.format(i)]['obs']['joint_states'])
        ee_pos = np.array(demo_data['demo_{}'.format(i)]['obs']['ee_pos'])
        ee_ori = np.array(demo_data['demo_{}'.format(i)]['obs']['ee_ori'])
        ee_state = np.array(demo_data['demo_{}'.format(i)]['obs']['ee_states'])

        gripper_state = np.zeros_like(action[:, -1])
        gripper_state[1:] = action[:-1, -1]
        gripper_state[0] = action[0, -1]

        gripper_position = np.array(demo_data['demo_{}'.format(i)]['obs']['gripper_states'])
        gripper_command = action[:, -1]

        # task emb
        # task_emb = self.lang_model.encode(language_instructions)

        # get episode length
        num_steps = obs.shape[0]
        
        episode_dir = episode_dir/str(episode_index).zfill(6)
        episode_dir.mkdir(exist_ok=True)

        # save episode length and language instruction
        with h5py.File(f'{episode_dir}/meta_info.h5', 'w') as h5_file:
            h5_file.create_dataset(name='length', data=num_steps)
        
        steps_dir = episode_dir/'steps'
        steps_dir.mkdir(exist_ok=True)
        for step_index in range(num_steps):
            step_dir = episode_dir/'steps'/str(step_index).zfill(4)
            step_dir.mkdir(exist_ok=True)
            
            with h5py.File(f'{step_dir}/other.h5', 'w') as h5_file:
                # language instruction
                h5_file.create_dataset('language_instruction', data=np.array(language_instructions, dtype=h5py.string_dtype(encoding='utf-8')))
                # task emb
                # h5_file.create_dataset(name='task_emb', data=task_emb)

                # episode length
                h5_file.create_dataset(name='episode_length', data=num_steps)

                # action
                h5_file.create_dataset(name='action', data=action[step_index])

                # observation (timestep, proprio, image_XXX)
                observation_group = h5_file.create_group(name='observation')

                ## image
                # ### image_primary
                Image.fromarray(obs[step_index]).save(f'{step_dir}/image_primary.jpg')
                ### image_wrist
                Image.fromarray(obs_wrist[step_index]).save(f'{step_dir}/image_wrist.jpg')

                ## proprio
                observation_group.create_dataset(name='proprio', data=joint_state[step_index])

                ## tcp_pose
                observation_group.create_dataset(name='tcp_pose', data=ee_state[step_index])

                ## gripper state (-1 or 1)
                observation_group.create_dataset(name='gripper_state', data=gripper_state[step_index])

                ## gripper position (n, 2)
                observation_group.create_dataset(name='gripper_position', data=gripper_position[step_index])

    def convert_origin_dataset_to_target(self, dataset_by_task):
        # /dataset_0
        # |_meta_info.h5
        # |_/episodes
        # | |_/0
        # | | |_/steps
        # | |   |_/0
        # |     | |_other.h5
        # |     | |_XXX.jpg
        # |     |...
        # | |_/1
        # | |_...
        # /dataset_1
        # |
        episodes_dir = self.tgt_dir/'episodes'
        episodes_dir.mkdir(exist_ok=True)

        num_episodes = 0
        for dataset in dataset_by_task:
            num_episodes += dataset['num_episode']

        if self.rank == 0:
            with h5py.File(f'{str(self.tgt_dir)}/meta_info.h5', 'w') as h5_file:
                h5_file.create_dataset(name='num_episodes', data=num_episodes)

        processed_task_num_episode = 0
        task_index = 0

        for episode_index in range(num_episodes):
            episode_index_in_task = episode_index - processed_task_num_episode

            if episode_index < self.start_episode_idx:
                continue
            if self.end_episode_idx is not None:
                if episode_index >= self.end_episode_idx:
                    break
            if episode_index % self.num_worker != self.rank:
                if episode_index_in_task+1 == dataset_by_task[task_index]['num_episode']:
                    processed_task_num_episode += dataset_by_task[task_index]['num_episode']
                    task_index += 1
                continue
            print(self.rank, episode_index, '/' , num_episodes)
            self.process_episode(episode_dir=episodes_dir, language_instructions=dataset_by_task[task_index]['language'], demo_data=dataset_by_task[task_index]['data'], episode_index=episode_index, episode_index_in_task=episode_index_in_task)

            if episode_index_in_task+1 == dataset_by_task[task_index]['num_episode']:
                processed_task_num_episode += dataset_by_task[task_index]['num_episode']
                task_index += 1

    def run(self):
        print(f'target dir: {self.tgt_dir}')

        dataset_by_task = []
        for path in list(Path(self.src_dir).iterdir()):
            path_name = str(path)
            task_name, task_language = extract_task_information(path_name, self.src_dir)
            demo_data = h5py.File(path_name, 'r')['data']
            num_episode = len(demo_data)
            dataset = {
                'language': task_language,
                'num_episode': num_episode,
                'data': demo_data
            }
            dataset_by_task.append(dataset)

        self.convert_origin_dataset_to_target(dataset_by_task)

        print(f'data saved at {self.tgt_dir}')

        # get data_info.json
        data_info = []
        episode_idx = 0
        total_step = 0
        for path in list(Path(self.src_dir).iterdir()):
            path_name = str(path)
            demo_data = h5py.File(path_name, 'r')['data']
            num_episode = len(demo_data)
            for i in range(num_episode):
                num_steps = np.array(demo_data['demo_{}'.format(i)]['obs']['agentview_rgb']).shape[0]
                data_info.append([str(episode_idx).zfill(6), num_steps])
                episode_idx += 1
                total_step += num_steps
        # print(total_step)
        with open(f'./data_info/{dataset_name}_converted.json', 'w') as f:
            json.dump(data_info, f)

def main(rank, port, num_worker, start_episode_idx=0, end_episode_idx=None):
    if num_worker > 1:
        setup(rank, world_size=num_worker, port=port)

    global dataset_name
    dataset_name = "libero_object" # "libero_10"
    src_dir = f"/inspire/hdd/global_user/guchun-240107140023/LIBERO/{dataset_name}"
    # tgt_dir = Path(f"/inspire/hdd/global_user/guchun-240107140023/LIBERO/{dataset_name}_converted")
    tgt_dir = Path(f"./{dataset_name}_converted")
    tgt_dir.mkdir(exist_ok=True) 

    dataset_converter = DatasetConverter(
        src_dir=src_dir,
        tgt_dir=tgt_dir,
        rank=rank,
        num_worker=num_worker,
        start_episode_idx=start_episode_idx,  # the dataset[start_episode_idx] will be processed
        end_episode_idx=end_episode_idx,  # None means the last episode. if not none, the dataset[end_episode_idx - 1] will be processed and the dataset[end_episode_idx] will not be processed
    )
    dataset_converter.run()

if __name__ == '__main__':
    start_episode_idx = 0
    end_episode_idx = None
    num_worker = 8
    port = (random.randint(0, 3000) % 3000) + 27000

    assert num_worker > 1
    mp.spawn(main, args=(port, num_worker, start_episode_idx, end_episode_idx), nprocs=num_worker, join=True)

    # main(0, port, 1, start_episode_idx=start_episode_idx, end_episode_idx=end_episode_idx)