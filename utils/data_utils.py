import logging

import numpy as np
try:
    from pytorch3d.transforms import (
        euler_angles_to_matrix,
        matrix_to_euler_angles,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
except:
    print('no pytorch3d')
import torch
from torch.cuda.amp import autocast
logger = logging.getLogger(__name__)
import functools
import math
import io
import os
import random
import re
import pickle
from multiprocessing import Value
from functools import partial
import json
from itertools import chain
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import lmdb
import numpy as np
from PIL import Image
import copy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from petrel_client.client import Client
except:
    pass 
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import bisect
from itertools import accumulate
import copy
from typing import List
from torchvision import transforms as torchtransforms
from PIL import Image
import clip
from pdb import set_trace
import h5py
from scipy.spatial.transform import Rotation as R
import time

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

MIN_KB = 10
MAX_NUM_IMAGES = 5
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Callable, Union

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
)

prop_state = DictConfig(
    {
        "n_scene_obs": 24,
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)

def _6d_to_pose(pose6d, degrees=False):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose

def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] =  R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d

def get_state_info_dict(episode: Dict[str, np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create a dictionary with raw state observations for environment resets.

    Args:
        episode: Sequence dictionary.

    Returns:
         Info dict of full robot and scene state (for env resets).
    """
    return {
        "state_info": {
            "robot_obs": torch.from_numpy(episode["robot_obs"]),
            "scene_obs": torch.from_numpy(episode["scene_obs"]),
        }
    }

def process_state(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    proprio_state: DictConfig,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    state_obs_keys = observation_space["state_obs"]
    state_obs_list_normalized = []
    state_obs_list_unnormalized = []
    for state_ob in state_obs_keys:
        if window_size == 0 and seq_idx == 0:  # single file loader
            state_tensor = torch.from_numpy(episode[state_ob]).float()
        else:  # episode loader
            state_tensor = torch.from_numpy(episode[state_ob][seq_idx : seq_idx + window_size]).float()
        # expand dims for single environment obs
        if len(state_tensor.shape) != 2:
            state_tensor = state_tensor.unsqueeze(0)
        # shape: (BxN_state_obs)
        assert len(state_tensor.shape) == 2
        if state_ob in transforms:
            state_tensor_normalized = transforms[state_ob](state_tensor)
            state_obs_list_normalized.append(state_tensor_normalized)
        else:
            state_obs_list_normalized.append(state_tensor)
        state_obs_list_unnormalized.append(state_tensor)
    seq_state_obs = torch.cat(state_obs_list_normalized, dim=1)
    seq_state_obs_unnormalized = torch.cat(state_obs_list_unnormalized, dim=1)

    if not proprio_state.normalize_robot_orientation and "robot_orientation_idx" in proprio_state:
        seq_state_obs[:, slice(*proprio_state.robot_orientation_idx)] = seq_state_obs_unnormalized[
            :, slice(*proprio_state.robot_orientation_idx)
        ]

    if not proprio_state.normalize:
        seq_state_obs = seq_state_obs_unnormalized

    # slice the specified parts of the proprioception state
    state_obs_sliced = []
    for slice_ids in proprio_state.keep_indices:
        seq_state_obs_ = seq_state_obs[:, slice(*slice_ids)]
        state_obs_sliced.append(seq_state_obs_)
    seq_state_obs = torch.cat(state_obs_sliced, dim=1)

    return {"robot_obs": seq_state_obs}

def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image

def preprocess_text_calvin(sample, tokenizer):
    text = tokenizer.tokenize(sample, truncate=True)
    return text

def process_rgb(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rgb_obs_keys = observation_space["rgb_obs"]

    seq_rgb_obs_dict = {}
    for _, rgb_obs_key in enumerate(rgb_obs_keys):
        rgb_obs = episode[rgb_obs_key]
        # expand dims for single environment obs
        if len(rgb_obs.shape) != 4:
            rgb_obs = np.expand_dims(rgb_obs, axis=0)
        assert len(rgb_obs.shape) == 4
        if window_size == 0 and seq_idx == 0:  # single file loader
            # To Square image
            seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte().permute(0, 3, 1, 2)
        else:  # episode loader
            seq_rgb_obs_ = torch.from_numpy(rgb_obs[seq_idx : seq_idx + window_size]).byte().permute(0, 3, 1, 2)
        # we might have different transformations for the different cameras
        if rgb_obs_key in transforms:
            seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
        seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    # shape: N_rgb_obs x (BxCxHxW)
    return {"rgb_obs": seq_rgb_obs_dict}

def process_depth(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # expand dims for single environment obs
    def exp_dim(depth_img):
        if len(depth_img.shape) != 3:
            depth_img = np.expand_dims(depth_img, axis=0)
        return depth_img

    depth_obs_keys = observation_space["depth_obs"]
    seq_depth_obs_dict = {}
    for _, depth_obs_key in enumerate(depth_obs_keys):
        depth_ob = exp_dim(episode[depth_obs_key])
        assert len(depth_ob.shape) == 3
        if window_size == 0 and seq_idx == 0:  # single file loader
            depth_ob_ = torch.from_numpy(depth_ob).float()
        else:  # episode loader
            depth_ob_ = torch.from_numpy(depth_ob[seq_idx : seq_idx + window_size]).float()
        # we might have different transformations for the different cameras
        if depth_obs_key in transforms:
            depth_ob_ = transforms[depth_obs_key](depth_ob_)
        seq_depth_obs_dict[depth_obs_key] = depth_ob_
    # shape: N_depth_obs x(BxHxW)
    return {"depth_obs": seq_depth_obs_dict}

def process_actions(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    # shape: (N_actions)
    action_keys = observation_space["actions"]
    if len(action_keys) != 1:
        raise NotImplementedError
    action_key = action_keys[0]
    if window_size == 0 and seq_idx == 0:  # single file loader
        action = episode[action_key]
        if "actions" in transforms:
            action = transforms["actions"]((action, episode["robot_obs"]))
        seq_acts = torch.from_numpy(action).float()
    else:  # episode loader
        seq_acts = torch.from_numpy(episode[action_key][seq_idx : seq_idx + window_size]).float()
    return {"actions": seq_acts}

def process_language(episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool) -> Dict[str, torch.Tensor]:
    seq_lang = {"lang": torch.empty(0)}
    if with_lang:
        lang = torch.from_numpy(episode["language"]).float()
        if "language" in transforms:
            lang = transforms["language"](lang)
        seq_lang["lang"] = lang
    return seq_lang

def lookup_naming_pattern(dataset_dir: Path, save_format: str) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits

def load_partial_traj_data():
    with open('utils/partial_task_data.json', 'r') as f:
        data = json.load(f)
    return data

def subtract_ranges(rangeA, rangeB):
    def subtract_single_range(a, b):
        result = []
        a_start, a_end = a
        b_start, b_end = b

        if b_start > a_end or b_end < a_start:
            # No overlap
            return [a]
        if b_start > a_start:
            result.append((a_start, min(a_end, b_start - 1)))
        if b_end < a_end:
            result.append((max(a_start, b_end + 1), a_end))

        return [r for r in result if r[0] <= r[1]]

    result = rangeA
    for b in rangeB:
        new_result = []
        for a in result:
            new_result.extend(subtract_single_range(a, b))
        result = new_result

    return result

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        *args: Any,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        pred_num: int=1,
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=False,
        text_aug=False,
        dif_ws=False,
        act_step=1,
        data_in_ceph=False,
        **kwargs: Any,
    ):
        self.load_dino_features = kwargs.get('load_dino_features', False)
        self.dino_features_path = kwargs.get('dino_features_path', None)
        self.load_sam_features = kwargs.get('load_sam_features', False)
        self.sam_features_path = kwargs.get('sam_features_path', None)
        self.load_track_labels = kwargs.get('load_track_labels', False)
        self.track_label_path = kwargs.get('track_label_path', None)
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.except_lang = key == "except_lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pred_num = pred_num
        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1 + self.pred_num - 1
            self.max_window_size = window_size + act_step - 1 + self.pred_num - 1
        else:
            self.min_window_size = min_window_size
            self.max_window_size = max_window_size
        self.act_step = act_step
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons
        self.data_in_ceph = data_in_ceph
        if self.data_in_ceph:
            self.conf_path = '~/petreloss.conf'
            self.client = Client(self.conf_path)
       
        with open('./utils/enrich_lang_annotations.json', 'r') as f:
            self.enrich_lang = json.load(f)
        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        if self.data_in_ceph:
            assert (
                "validation" in self.abs_datasets_dir
                or "training" in self.abs_datasets_dir
            )
            self.validation = "validation" in self.abs_datasets_dir
        else:
            assert (
                "validation" in self.abs_datasets_dir.as_posix()
                or "training" in self.abs_datasets_dir.as_posix()
            )
            self.validation = "validation" in self.abs_datasets_dir.as_posix()
        print(f"loading dataset at {self.abs_datasets_dir}")
        print("finished loading dataset")

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}


    def process_features(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
        obs_key: str = 'dino_features_obs'
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        obs_space = {
            "dino_features_obs": ['dino_feats_static', 'dino_feats_gripper'],
            "sam_features_obs": ['sam_feats_static', 'sam_feats_gripper']
        }
        rgb_obs_keys = obs_space[obs_key]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 3:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 3
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).float()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).float()
            
            #if rgb_obs_key in transforms:
            #    seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {obs_key: seq_rgb_obs_dict}

    def process_track_label(
        self,
        episode: Dict[str, np.ndarray],
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        seq_track_label_dict = {}
        tracks = episode['traj_static'] # [Nframe, Npoints, 2]
        visibility = episode['visibility_static'] # [Nframe, Npoints]
        if window_size == 0 and seq_idx == 0:
            tracks = torch.from_numpy(tracks)
            visibility = torch.from_numpy(visibility)
        else:
            tracks = torch.from_numpy(tracks[seq_idx : seq_idx + window_size])
            visibility = torch.from_numpy(visibility[seq_idx : seq_idx + window_size])

        tracks_gripper = episode['traj_gripper'] # [Nframe, Npoints, 2]
        visibility_gripper = episode['visibility_gripper'] # [Nframe, Npoints]
        if window_size == 0 and seq_idx == 0:
            tracks_gripper = torch.from_numpy(tracks_gripper)
            visibility_gripper = torch.from_numpy(visibility_gripper)
        else:
            tracks_gripper = torch.from_numpy(tracks_gripper[seq_idx : seq_idx + window_size])
            visibility_gripper = torch.from_numpy(visibility_gripper[seq_idx : seq_idx + window_size])

        seq_track_label_dict['tracks'] = tracks
        seq_track_label_dict['track_visibility'] = visibility
        seq_track_label_dict['tracks_gripper'] = tracks_gripper
        seq_track_label_dict['track_visibility_gripper'] = visibility_gripper
        return {"track_label": seq_track_label_dict}
    

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]} if with_lang else {"lang": 'no lang'}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = self._get_window_size(idx)
            else:
                print(
                    f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx
        
        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)
        
        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list
        
        
        
        
         # wyzhang
        if sequence['depth_obs'] is not None:
            new_depth_list = []
            np_depth = copy.deepcopy(sequence["depth_obs"]["depth_static"].numpy())
            for i in range(np_depth.shape[0]):
                new_depth_list.append(np_depth[i, :, :])
            sequence["depth_obs"]["depth_static"] = new_depth_list   
            
            
            new_list = []
            np_depth_gripper = copy.deepcopy(sequence["depth_obs"]["depth_gripper"].numpy())
            for i in range(np_gripper.shape[0]):
                new_list.append(np_depth_gripper[i, :, :])
            sequence["depth_obs"]["depth_gripper"] = new_list     
        
        
        
        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        if self.load_dino_features:
            seq_dino_obs = self.process_features(episode, self.observation_space, self.transforms, obs_key='dino_features_obs')
            # seq_dino_obs = dict()
        if self.load_sam_features:
            seq_sam_obs = self.process_features(episode, self.observation_space, self.transforms, obs_key='sam_features_obs')
            # seq_sam_obs = dict()
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        if self.load_track_labels:
            seq_track_label = self.process_track_label(episode)
        info = get_state_info_dict(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
            **(seq_track_label if self.load_track_labels else dict()),
            **(seq_dino_obs if self.load_dino_features else dict()),
            **(seq_sam_obs if self.load_sam_features else dict()),
        }  
        seq_dict["idx"] = idx  

        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif (
            self.episode_lookup[idx + window_diff]
            != self.episode_lookup[idx] + window_diff
        ):
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(
                self.max_window_size, (self.min_window_size + steps_to_next_episode - 1)
            )
        else:
            max_window = self.max_window_size

        return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        if 'track_label' in seq:
            seq.update(
                {
                    "track_label": {
                        k: self._pad_with_repetition(v, pad_size, head)
                        for k, v in seq["track_label"].items()
                    }
                }
            )
        if 'dino_features_obs' in seq:
            seq.update(
                {
                    "dino_features_obs": {
                        k: self._pad_with_repetition(v, pad_size, head)
                        for k, v in seq["dino_features_obs"].items()
                    }
                }
            )
        if 'sam_features_obs' in seq:
            seq.update(
                {
                    "sam_features_obs": {
                        k: self._pad_with_repetition(v, pad_size, head)
                        for k, v in seq["sam_features_obs"].items()
                    }
                }
            )

        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

class DiskCalvinDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        text_fn: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = self.load_pkl
        elif self.save_format == "npz":
            self.load_file = partial(self.load_npz, data_in_ceph=self.data_in_ceph)
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_lookup,
                self.lang_ann,
                self.lang_task
            ) = self._build_file_indices_lang()
        elif self.except_lang:
            self.episode_lookup = self._build_file_indices_except_lang()
        else:
            self.episode_lookup = self._build_file_indices()

        if self.data_in_ceph:
            self.naming_pattern, self.n_digits = self.ceph_lookup_naming_pattern()
        else:
            self.naming_pattern, self.n_digits = lookup_naming_pattern(
                self.abs_datasets_dir, self.save_format
            )
            
        self.cache_static = None
        self.cache_gripper = None
        # self._open_envs()

    # def _open_envs(self):
    #     import lmdb
    #     self.dino_static_env = lmdb.open('calvin_dino_static_training.lmdb', readonly=True, lock=False, readahead=False, max_readers=32)
    #     self.dino_gripper_env = lmdb.open('calvin_dino_gripper_training.lmdb', readonly=True, lock=False, readahead=False, max_readers=32)
        
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state['dino_static_env'] = None
    #     state['dino_gripper_env'] = None
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     self._open_envs()

    def ceph_lookup_naming_pattern(self):
        filenames = self.client.list(self.abs_datasets_dir)
        for filename in filenames:
            if self.save_format in filename:
                break
        filename = self.abs_datasets_dir + f"/{filename}"
        suffix = "." + self.save_format
        stem_suffix = filename.split('/')[-1]
        stem = stem_suffix.replace(suffix, "")
        aux_naming_pattern = re.split(r"\d+", stem)
        naming_pattern = (filename.replace(stem_suffix, aux_naming_pattern[0]), suffix)
        n_digits = len(re.findall(r"\d+", stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        if self.data_in_ceph:
            return f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        else:
            return Path(
                f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
            )

    def _load_dino_feat(self, file_idx, img_key):
        return (torch.load(os.path.join(self.dino_features_path, img_key, 'training' if not self.validation else 'validation', f'{file_idx}.pt')).to(torch.float32)).numpy()

    def _load_sam_feat(self, file_idx, img_key):
        return (torch.load(os.path.join(self.sam_features_path, img_key, 'training' if not self.validation else 'validation', f'{file_idx}.pt')).to(torch.float32)).numpy()

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # if self.cache:
        #     return self.cache
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        if self.load_dino_features:
            keys.append("dino_static")
            keys.append("dino_gripper")
        if self.load_sam_features:
            keys.append("sam_static")
            keys.append("sam_gripper")
        if self.load_track_labels:
            keys.append("traj_static")
            keys.append("traj_gripper")
            keys.append("visibility_static")
            keys.append("visibility_gripper")
            
        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in range(start_idx, end_idx)
        ]
        # episode_names = [self._get_episode_name(file_idx) for file_idx in range(start_idx, end_idx)]
        # with ThreadPoolExecutor(max_workers=64) as executor:
        #     episodes = list(executor.map(self.load_file, episode_names))

        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang

        # lhs: dino & sam features
        # start_time = time.time()
        if self.load_dino_features:
            # assert self.dino_features_path is not None
            # if self.cache_static is None:
            #     ################### vanilla #######################
            #     dino_feats_static = []
            #     dino_feats_gripper = []
            #     for file_idx in range(start_idx, end_idx):
            #         feat_static = (torch.load(os.path.join(self.dino_features_path, 'rgb_static', 'training' if not self.validation else 'validation', f'{file_idx}.pt')).to(torch.float32)).numpy()
            #         dino_feats_static.append(feat_static)
            #         feat_gripper = (torch.load(os.path.join(self.dino_features_path, 'rgb_gripper', 'training' if not self.validation else 'validation', f'{file_idx}.pt')).to(torch.float32)).numpy()
            #         dino_feats_gripper.append(feat_gripper)   
            #     self.cache_gripper = dino_feats_gripper
            #     episode['dino_feats_static'] = np.stack(dino_feats_static)
            #     self.cache_static = episode['dino_feats_static']
            #     episode['dino_feats_gripper'] = np.stack(dino_feats_gripper)
            #     self.cache_gripper =  episode['dino_feats_gripper']
            # else:
            #     episode['dino_feats_static'] = self.cache_static 
            #     episode['dino_feats_gripper'] = self.cache_gripper 

            ################## multithread #######################
            # _func = partial(self._load_dino_feat, img_key='rgb_static')
            # with ThreadPoolExecutor(max_workers=16) as executor:
            #     dino_feats_static = list(executor.map(_func, list(range(start_idx, end_idx))))

            # _func = partial(self._load_dino_feat, img_key='rgb_gripper')
            # with ThreadPoolExecutor(max_workers=16) as executor:
            #     dino_feats_gripper = list(executor.map(_func, list(range(start_idx, end_idx))))
            # episode['dino_feats_static'] = np.stack(dino_feats_static)
            # episode['dino_feats_gripper'] = np.stack(dino_feats_gripper)

            ################## npz #######################
            # for file_idx in range(start_idx, end_idx):
            #     feat = (np.load(os.path.join(self.dino_features_path, 'training' if not self.validation else 'validation', f'{file_idx}.npz')))
            #     dino_feats_static.append(feat['static'])
            #     dino_feats_gripper.append(feat['gripper'])   

            #### zwy ## 
            # _get_dino_name = lambda file_idx: Path(os.path.join(self.dino_features_path, 'training/training' if not self.validation else 'validation', f'{file_idx}{self.naming_pattern[1]}'))
            # dino_labels = [self.load_file(_get_dino_name(file_idx)) for file_idx in range(start_idx, end_idx)]     
            # try:
            #     dino_feats_static = [ep['static'] for ep in dino_labels]
            #     dino_feats_gripper =  [ep['gripper'] for ep in dino_labels]
            # except:
            #     print('static is not a file in the archive', f'start_idx.{start_idx}', f'end_idx.{end_idx}')
            ################### lmdb #######################
            # with self.dino_static_env.begin() as txn:
            #     for file_idx in range(start_idx, end_idx):
            #         raw = txn.get(f'{file_idx}'.encode())
            #         data = pickle.loads(raw)
            #         dino_feats_static.append(data.to(torch.float32).numpy())
            # with self.dino_gripper_env.begin() as txn:
            #     for file_idx in range(start_idx, end_idx):
            #         raw = txn.get(f'{file_idx}'.encode())
            #         data = pickle.loads(raw)
            #         dino_feats_gripper.append(data.to(torch.float32).numpy())

            ################### from episode #######################
            episode['dino_feats_static'] = episode['dino_static']
            episode['dino_feats_gripper'] = episode['dino_gripper']
            episode.pop('dino_static', None)
            episode.pop('dino_gripper', None)


        # if time.time()-start_time > 0.3:
            # print("dino_time", time.time()-start_time)
        # start_time = time.time()
        if self.load_sam_features:
            sam_feats_static = []
            sam_feats_gripper = []
            # assert self.sam_features_path is not None
            ################### vanilla #######################
            # for file_idx in range(start_idx, end_idx):
            #     feat_static = (torch.load(os.path.join(self.sam_features_path, 'rgb_static', 'training' if not self.validation else 'validation', f'{file_idx}.pt')).to(torch.float32)).numpy()
            #     sam_feats_static.append(feat_static)
            #     feat_gripper = (torch.load(os.path.join(self.sam_features_path, 'rgb_gripper', 'training' if not self.validation else 'validation', f'{file_idx}.pt')).to(torch.float32)).numpy()
            #     sam_feats_gripper.append(feat_gripper)

            ################## multithread #######################
            # _func = partial(self._load_sam_feat, img_key='rgb_static')
            # _file_indices = list(range(start_idx, end_idx))
            # with ThreadPoolExecutor(max_workers=16) as executor:
            #     sam_feats_static = list(executor.map(_func, _file_indices))

            # _func = partial(self._load_sam_feat, img_key='rgb_gripper')
            # with ThreadPoolExecutor(max_workers=16) as executor:
            #     sam_feats_gripper = list(executor.map(_func, _file_indices))
            # episode['sam_feats_static'] = np.stack(sam_feats_static).transpose(0, 2, 1)
            # episode['sam_feats_gripper'] = np.stack(sam_feats_gripper).transpose(0, 2, 1)

            ################### from episode #######################


            # lhs: 注意，存的时候是把[B,C,H,W]弄成了[B,C,HW]存的，所以token维度在后面，这里保持和dino feature一致
            episode['sam_feats_static'] = episode['sam_static'].transpose(0, 2, 1)
            episode['sam_feats_gripper'] = episode['sam_gripper'].transpose(0, 2, 1)
            episode.pop('sam_static', None)
            episode.pop('sam_gripper', None)

        # if time.time()-start_time > 0.3:
            # print("sam_time", time.time()-start_time)
        # lhs: cotracker label
        # start_time = time.time()
        
        
        # if self.load_track_labels:
        #     # rgb static
        #     _get_track_name = lambda file_idx: Path(os.path.join(self.track_label_path, 'rgb_static', 'training' if not self.validation else 'validation', f'{file_idx}{self.naming_pattern[1]}'))
        #     track_labels = [self.load_file(_get_track_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        #     track_keys = ['tracks', 'visibility']           
        #     tracks = {key: np.stack([ep[key] for ep in track_labels]) for key in track_keys}
            
        #     # rgb gripper
        #     _get_track_name = lambda file_idx: Path(os.path.join(self.track_label_path, 'rgb_gripper', 'training' if not self.validation else 'validation', f'{file_idx}{self.naming_pattern[1]}'))
        #     track_labels = [self.load_file(_get_track_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        #     track_keys = ['tracks', 'visibility']
        #     _tracks = {key: np.stack([ep[key] for ep in track_labels]) for key in track_keys}
        #     tracks_gripper = {}
        #     for k, v in _tracks.items():
        #         tracks_gripper[k+"_gripper"] = v

        #     episode = {**episode, **tracks, **tracks_gripper}
        
        
        # if time.time()-start_time > 0.1:
            # print("traj_time", time.time()-start_time)
        return episode

    def _build_file_indices_lang(
        self, # abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        abs_datasets_dir = self.abs_datasets_dir
        episode_lookup = []

        try:
            if self.data_in_ceph:
                print(
                "trying to load lang data from: ",
                abs_datasets_dir +f"/{self.lang_folder}/auto_lang_ann.npy",
                )
                lang_data_bytes = self.client.get(abs_datasets_dir+f"/{self.lang_folder}/auto_lang_ann.npy", enable_cache=True)
                lang_data = io.BytesIO(lang_data_bytes)
                lang_data = np.load(lang_data, allow_pickle=True).item()
            else:
                print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                )
                lang_data = np.load(
                    abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                    allow_pickle=True,
                ).item()
        except Exception:
            if self.data_in_ceph:
                print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir + "/auto_lang_ann.npy",
                )
                lang_data_bytes = self.client.get(abs_datasets_dir+f"/auto_lang_ann.npy", enable_cache=True)
                lang_data = io.BytesIO(lang_data_bytes)
                lang_data = np.load(lang_data, allow_pickle=True).item()
            else:
                print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
                )
                lang_data = np.load(
                    abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
                ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        partial_st_ed_list = load_partial_traj_data()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.partial_data:
                if [start_idx, end_idx] not in partial_st_ed_list:
                    continue
            if self.pretrain:
                start_idx = max(
                    start_idx,
                    end_idx + 1 - self.min_window_size - self.aux_lang_loss_window,
                )
            assert end_idx >= self.max_window_size
            cnt = 0
            
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _build_file_indices(self) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        abs_datasets_dir = self.abs_datasets_dir
        episode_lookup = []

        if self.data_in_ceph:
            lang_data_bytes = self.client.get(abs_datasets_dir+f"ep_start_end_ids.npy", enable_cache=True)
            lang_data = io.BytesIO(lang_data_bytes)
            ep_start_end_ids = np.load(lang_data)
        else:
            ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def _build_file_indices_except_lang(self) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        abs_datasets_dir = self.abs_datasets_dir
        # lang_data = np.load(
        #     abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
        #     allow_pickle=True,
        # ).item()
        lang_data = np.load(
            # abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            "data_info/auto_lang_ann.npy",
            allow_pickle=True,
        ).item()
        lang_ep_start_end_ids = lang_data["info"]["indx"]

        episode_lookup = []

        if self.data_in_ceph:
            lang_data_bytes = self.client.get(abs_datasets_dir+f"ep_start_end_ids.npy", enable_cache=True)
            lang_data = io.BytesIO(lang_data_bytes)
            ep_start_end_ids = np.load(lang_data)
        else:
            ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        ep_start_end_ids = np.load(abs_datasets_dir / "except_lang_idx" / "except_lang_idx.npy").tolist() # if there is bug, replace it with "ep_start_end_ids = np.load(abs_datasets_dir /  "ep_start_end_ids.npy").tolist()"

        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def collator(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        depth_static_tensors = torch.stack([depth_image_fn(s["depth_obs"]["depth_static"]) for s in sample])
        depth_gripper_tensors =  torch.stack([depth_image_fn(s["depth_obs"]["depth_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        text_tensors = self.text_fn(stacked_language)

        if 'track_label' in sample[0]:
            track_tensors = torch.stack([s["track_label"]["tracks"] for s in sample])
            track_vis_tensors = torch.stack([s["track_label"]["track_visibility"] for s in sample])
            track_gripper_tensors = torch.stack([s["track_label"]["tracks_gripper"] for s in sample])
            track_vis_gripper_tensors = torch.stack([s["track_label"]["track_visibility_gripper"] for s in sample])
        
        dino_features_tensors = None
        dino_features_gripper_tensors = None
        sam_features_tensors = None
        sam_features_gripper_tensors = None
        if 'dino_features_obs' in sample[0]:
            dino_features_tensors = torch.stack([s["dino_features_obs"]["dino_feats_static"] for s in sample])
            dino_features_gripper_tensors = torch.stack([s["dino_features_obs"]["dino_feats_gripper"] for s in sample])

        if 'sam_features_obs' in sample[0]:
            sam_features_tensors = torch.stack([s["sam_features_obs"]["sam_feats_static"] for s in sample])
            sam_features_gripper_tensors = torch.stack([s["sam_features_obs"]["sam_feats_gripper"] for s in sample])


        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
                depth_static_tensors = self.rgb_shift.forward_traj(depth_static_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
                depth_gripper_tensors = self.gripper_shift.forward_traj(depth_gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        

        
        robot_obs = torch.zeros(1)
        
        if self.act_step != 1:
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]
            
            depth_static_tensors = depth_static_tensors[:, :-(self.act_step-1)]
            depth_gripper_tensors = depth_gripper_tensors[:, :-(self.act_step-1)]

            if 'track_label' in sample[0]:
                track_tensors = track_tensors[:, :-(self.act_step-1)]
                track_vis_tensors = track_vis_tensors[:, :-(self.act_step-1)]
                track_gripper_tensors = track_gripper_tensors[:, :-(self.act_step-1)]
                track_vis_gripper_tensors = track_vis_gripper_tensors[:, :-(self.act_step-1)]
            if 'dino_features_obs' in sample[0]:
                dino_features_tensors = dino_features_tensors[:, :-(self.act_step-1)]
                dino_features_gripper_tensors = dino_features_gripper_tensors[:, :-(self.act_step-1)]
            if 'sam_features_obs' in sample[0]:
                sam_features_tensors = sam_features_tensors[:, :-(self.act_step-1)]
                sam_features_gripper_tensors = sam_features_gripper_tensors[:, :-(self.act_step-1)]
            
            
            
            # fwd_chunk
            # image_tensors_chunk = image_tensors.unfold
            
            
        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs, depth_static_tensors, depth_gripper_tensors, dino_features_tensors, dino_features_gripper_tensors, sam_features_tensors, sam_features_gripper_tensors, \
        dict(tracks=track_tensors, track_visibility=track_vis_tensors, tracks_gripper=track_gripper_tensors, track_visibility_gripper=track_vis_gripper_tensors) if self.load_track_labels else dict(),

    def load_pkl(self, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def load_npz(self, filename, data_in_ceph=False):
        if not data_in_ceph:
            return np.load(filename.as_posix())
        else:
            data_bytes = self.client.get(filename, enable_cache=True)
            data = io.BytesIO(data_bytes)
            try:
                data = np.load(data, allow_pickle=True)
            except:
                data = np.load(data)
            return data

def get_calvin_dataset(args, image_processor, tokenizer, epoch=0, floor=False, except_lang=False):
    dataset_path = args.calvin_dataset
    # ann is dict including language and info
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    if args.data_in_ceph:
        datasets_dir = dataset_path + "/training"
    else:
        datasets_dir = Path(dataset_path) / "training"
    calvin_dataset = DiskCalvinDataset(
        datasets_dir=datasets_dir,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        pred_num=args.pred_num,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data,
        data_in_ceph=args.data_in_ceph,
        key='except_lang' if except_lang else 'lang',
        load_track_labels=args.load_track_labels,
        track_label_path=args.track_label_path,
        load_dino_features=args.load_dino_features,
        dino_features_path=args.dino_features_path,
        load_sam_features=args.load_sam_features,
        sam_features_path=args.sam_features_path
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  #
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=calvin_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)

def get_calvin_val_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_path = args.calvin_dataset
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    if args.data_in_ceph:
        datasets_dir = dataset_path + "/validation"
    else:
        datasets_dir = Path(dataset_path) / "validation"
    calvin_dataset = DiskCalvinDataset(
        datasets_dir=datasets_dir,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data,
        data_in_ceph=args.data_in_ceph
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        shuffle=False,
        persistent_workers=True,
        collate_fn=calvin_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)

class BaseDroidDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        image_primary_size=200,
        image_wrist_size=84,
        obs_space: DictConfig = obs_config,
        proprio_state: DictConfig = prop_state,
        transforms: Dict = {},
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        text_aug=False,
        dif_ws=False,
        act_step: int = 1,
        key: str = "lang",
        language_mode: str = "language_instruction",
        primary_mode: str = "image_primary",
        dataset_info: str = "droid_success",
        small_size: int = 0,
        data_in_ceph: bool = False, 
        max_rel_pos: float = 0.02,
        max_rel_orn: float = 0.05,
        magic_scaling_factor_pos: float = 1.0,
        magic_scaling_factor_orn: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_name = dataset_name 
        self.dataset_info = dataset_info
        self.root_dir = root_dir 
        self.dataset_path = f'{root_dir}/{dataset_name}' 
        self.conf_path = '~/petreloss.conf'
        self.client = Client(self.conf_path)
        self.image_primary_size = image_primary_size
        self.image_wrist_size = image_wrist_size
        self.image_preprocess = None
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.window_size = window_size
        self.language_mode = language_mode
        self.primary_mode = primary_mode
        self.small_size = small_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            raise NotImplementedError
        
        assert self.max_window_size == self.min_window_size
        self.aux_lang_loss_window = aux_lang_loss_window
        self.text_aug = text_aug
        self.act_step = act_step
        self.data_in_ceph = data_in_ceph
        self.max_rel_pos = max_rel_pos
        self.max_rel_orn = max_rel_orn
        self.magic_scaling_factor_pos = magic_scaling_factor_pos
        self.magic_scaling_factor_orn = magic_scaling_factor_orn
        logger.info(f"loading dataset at {root_dir}/{dataset_name}")
        logger.info("finished loading dataset")
        if self.data_in_ceph:
            assert self.client.isdir(self.dataset_path)
        else:
            os.path.exists(self.dataset_path)
        assert os.path.exists(f"data_info/{self.dataset_info}.json")
        with open(f"data_info/{self.dataset_info}.json", 'r') as f: #TODO
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] - self.max_window_size for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)

        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_

        return {"rgb_obs": seq_rgb_obs_dict}

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))

        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))

        return padded

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int): 
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} != max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx

        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)

        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list

        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
        if episode_id - 1 >= 0:
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = idx
        num_step_per_episode = self.num_step_per_episode[episode_id]
        end_id = min(start_id + window_size, num_step_per_episode)
        episode_id = self.episode_list[episode_id]
        episodes = []
        for step_id in range(start_id, end_id):
            data_dict = {}
            try:
                step_id = str(step_id).zfill(4)
                other_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/other.h5"
                if self.data_in_ceph:
                    other_bytes = self.client.get(other_path, enable_cache=True)
                    other_h5_file = h5py.File(io.BytesIO(other_bytes))
                else:
                    other_h5_file = h5py.File(other_path)
            except:
                print("episode_id :", episode_id)
                print("step_id :", step_id)
                print("num_step_per_episode :", num_step_per_episode)
                print("other_path", f"{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/other.h5")
            data_dict["rgb_static"] = self.load_primary_rgb(episode_id, step_id, self.primary_mode)
            data_dict["rgb_gripper"] = self.load_wrist_rgb(episode_id, step_id)
            data_dict["rel_actions"] = self.load_action(other_h5_file)
            data_dict["robot_obs"] = self.load_robot_obs(other_h5_file)
            data_dict["scene_obs"] = self.load_scene_obs(episode_id, step_id)
            episodes.append(data_dict)
   
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode["language"] = self.load_language_instruction(other_h5_file, self.language_mode)
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        info["use_for_aux_lang_loss"] = False
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  
        seq_dict["droid_episode_id"] = episode_id

        return seq_dict

    def load_primary_rgb(self, episode_id, step_id, primary_mode="image_primary"):
        image_primary = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/{primary_mode}.jpg'
        if self.data_in_ceph:
            image_primary = self.client.get(image_primary, enable_cache=True)
            image_primary = io.BytesIO(image_primary)
        image_primary = np.array(Image.open(image_primary).convert("RGB"))

        return image_primary.astype(np.uint8)

    def load_wrist_rgb(self, episode_id, step_id):
        image_wrist = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/image_wrist.jpg'
        if self.data_in_ceph:
            image_wrist = self.client.get(image_wrist, enable_cache=True)
            image_wrist = io.BytesIO(image_wrist)
        image_wrist = np.array(Image.open(image_wrist).convert("RGB")) 

        return image_wrist.astype(np.uint8)

    def load_language_instruction(self, other_h5_file, language_mode="language_instruction"):
        if "full" not in self.dataset_info:
            language_instruction = other_h5_file[language_mode][()].decode('utf-8')
        else:
            language_instruction = "No language instruction."

        return language_instruction
        
    def load_action(self, other_h5_file, max_rel_pos=0.02, max_rel_orn=0.05, magic_scaling_factor_pos=1.0, magic_scaling_factor_orn=1.0):
        action = other_h5_file["action_delta_wrist_pose"][()]
        action[:3] /= (self.max_rel_pos * self.magic_scaling_factor_pos)
        action[3:6] /= (self.max_rel_orn * self.magic_scaling_factor_orn)

        return action

    def load_robot_obs(self, other_h5_file):
        robot_obs = np.zeros(self.proprio_state.n_state_obs)
        robot_obs[:6] = other_h5_file['observation']['gripper_pose6d'][()]
        robot_obs[-1] = other_h5_file['observation']['gripper_open_state'][()]
        robot_obs[7:14] = other_h5_file['observation']['joint_position'][()]

        return robot_obs

    def load_scene_obs(self, episode_id, step_id):
        scene_obs = np.zeros(self.proprio_state.n_scene_obs)

        return scene_obs

    def __len__(self):
        if self.small_size:
            return self.small_size
        else:
            return self.length

class DistDroidDataset(Dataset):
    def __init__(
        self, 
        image_fn: Callable,
        text_fn: Callable,
        dataset_names: List[str],
        *args: Any,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        traj_cons: bool = False,
        act_step : int = 1,
        small_size: int = 0, 
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.datasets = [
            BaseDroidDataset(
                *args, 
                dataset_name=dataset_name,
                act_step=act_step,
                small_size=small_size,
                **kwargs,
                
            ) for dataset_name in dataset_names
        ]
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.traj_cons = traj_cons
        self.act_step = act_step
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        self.length_each_dataset = [len(dataset) for dataset in self.datasets]
        self.accumulated_length_each_dataset = list(accumulate(self.length_each_dataset))

    def register_image_preprocess_hook(self, func):
        self.image_preprocess = func

    def __len__(self):
        return self.accumulated_length_each_dataset[-1]

    def __getitem__(self, idx):
        dataset_id = bisect.bisect_right(self.accumulated_length_each_dataset, idx)
        if dataset_id - 1 >= 0:
            local_idx = idx - self.accumulated_length_each_dataset[dataset_id - 1]
        else:
            local_idx = idx

        return self.datasets[dataset_id].__getitem__(local_idx)

    def collator(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        droid_episode_id = [s["droid_episode_id"] for s in sample]
        text_tensors = self.text_fn(stacked_language)
        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        robot_obs = torch.zeros(1)
        if self.act_step != 1:
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]

            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]

        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs 

def get_droid_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_names = ["droid_success"]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    droid_dataset = DistDroidDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        dataset_info=args.dataset_info,
        data_in_ceph=args.data_in_ceph,
        max_rel_pos=args.max_rel_pos,
        max_rel_orn=args.max_rel_orn,
        magic_scaling_factor_pos=args.magic_scaling_factor_pos,
        magic_scaling_factor_orn=args.magic_scaling_factor_orn,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(droid_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        droid_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        droid_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=droid_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=droid_dataset)

class BaseLiberoDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        image_primary_size=200,
        image_wrist_size=84,
        obs_space: DictConfig = obs_config,
        proprio_state: DictConfig = prop_state,
        transforms: Dict = {},
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        text_aug=False,
        dif_ws=False,
        act_step: int = 1,
        key: str = "lang",
        language_mode: str = "language_instruction",
        primary_mode: str = "image_primary",
        dataset_info: str = "libero",
        small_size: int = 0,
        gripper_width: bool = False,
        load_libero_file: str = "h5", 
        **kwargs: Any,
    ):
        super().__init__()
        
        self.dataset_name = dataset_name
        self.dataset_info = dataset_info
        self.root_dir = root_dir 
        self.dataset_path = f'{root_dir}/{dataset_name}' 
        self.conf_path = '~/petreloss.conf'
        self.image_primary_size = image_primary_size
        self.image_wrist_size = image_wrist_size
        self.image_preprocess = None
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.window_size = window_size
        self.language_mode = language_mode
        self.primary_mode = primary_mode
        self.small_size = small_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            raise NotImplementedError
        
        assert self.max_window_size == self.min_window_size
        self.aux_lang_loss_window = aux_lang_loss_window
        self.text_aug = text_aug
        self.act_step = act_step
        logger.info(f"loading dataset at {root_dir}/{dataset_name}")
        logger.info("finished loading dataset")
        assert os.path.exists(f"./data_info/{self.dataset_info}.json")
        with open(f"./data_info/{self.dataset_info}.json", 'r') as f:
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] - self.max_window_size for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)

        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]
        self.gripper_width = gripper_width
        self.load_libero_file = load_libero_file

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)

        return {"rgb_obs": seq_rgb_obs_dict}

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """

        return self.max_window_size - len(sequence["actions"])

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))

        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))

        return padded

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )

        return seq

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} != max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx

        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)

        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list

        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
        if episode_id - 1 >= 0:
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = idx
        num_step_per_episode = self.num_step_per_episode[episode_id]
        end_id = min(start_id + window_size, num_step_per_episode)
        episode_id = self.episode_list[episode_id]
        episodes = []
        for step_id in range(start_id, end_id):
            data_dict = {}
            str_step_id = str(step_id).zfill(4)
            if self.load_libero_file == "h5":
                other_path = f"{self.dataset_path}/episodes/{episode_id}/steps/{str_step_id}/other.h5"
                other_file = h5py.File(other_path)
            data_dict["rgb_static"] = self.load_primary_rgb(episode_id, str_step_id, self.primary_mode)
            data_dict["rgb_gripper"] = self.load_wrist_rgb(episode_id, str_step_id)
            data_dict["rel_actions"] = self.load_action(other_file)
            data_dict["robot_obs"] = self.load_robot_obs(other_file)
            data_dict["scene_obs"] = self.load_scene_obs(episode_id, str_step_id)
            episodes.append(data_dict)
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode["language"] = self.load_language_instruction(other_file, self.language_mode)
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        info["use_for_aux_lang_loss"] = False
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  
        seq_dict["episode_id"] = episode_id

        return seq_dict

    def load_primary_rgb(self, episode_id, step_id, primary_mode="image_primary"):
        image_primary_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/{primary_mode}.jpg'
        image_primary = np.array(Image.open(image_primary_path).convert("RGB"))

        return image_primary.astype(np.uint8)

    def load_wrist_rgb(self, episode_id, step_id):
        image_wrist_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/image_wrist.jpg'
        image_wrist = np.array(Image.open(image_wrist_path).convert("RGB"))

        return image_wrist.astype(np.uint8)

    def load_language_instruction(self, other_file, language_mode="language_instruction"):
        if self.load_libero_file == "h5":
            language_instruction = other_file[language_mode][()].decode('utf-8')
        elif self.load_libero_file == "npz":
            language_instruction = other_file[language_mode].tobytes().decode('utf-8')
        else:
            raise NotImplementedError

        return language_instruction
        
    def load_action(self, other_file, max_rel_pos=0.02, max_rel_orn=0.05, magic_scaling_factor_pos=1.0, magic_scaling_factor_orn=1.0):
        if self.load_libero_file == "h5":
            action = other_file["action"][()]
        elif self.load_libero_file == "npz":
            action = other_file["action"]
        else:
            raise NotImplementedError
        
        return action

    def load_robot_obs(self, other_file):
        robot_obs = np.zeros(self.proprio_state.n_state_obs)
        if self.load_libero_file == "h5":
            robot_obs[:6] = other_file['observation']['tcp_pose'][:6]
            euler = R.from_euler("xyz", robot_obs[3:6], degrees=False)
            euler = euler.as_euler("xyz", degrees=False)
            robot_obs[3:6] = euler
            robot_obs[-1] = other_file['observation']['gripper_state'][()]
            robot_obs[7:14] = other_file['observation']['proprio'][()]
            if self.gripper_width:
                robot_obs[-2:] = other_file['observation']['gripper_position'][()]
        elif self.load_libero_file == "npz":
            robot_obs[:6] = other_file["observation_tcp_pose"][:6]
            euler = R.from_euler("xyz", robot_obs[3:6], degrees=False)
            euler = euler.as_euler("xyz", degrees=False)
            robot_obs[3:6] = euler
            robot_obs[-1] = other_file["observation_gripper_state"]
            robot_obs[7:14] = other_file["observation_proprio"]
            if self.gripper_width:
                robot_obs[-2:] = other_file["observation_gripper_position"]
        else:
            raise NotImplementedError      

        return robot_obs

    def load_scene_obs(self, episode_id, step_id):
        scene_obs = np.zeros(self.proprio_state.n_scene_obs)

        return scene_obs

    def __len__(self):
        if self.small_size:
            return self.small_size
        else:
            return self.length

class DiskLiberoDataset(Dataset):
    def __init__(
        self, 
        image_fn: Callable,
        text_fn: Callable,
        dataset_names: List[str],
        *args: Any,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        traj_cons: bool = False,
        act_step : int = 1,
        small_size: int = 0, 
        gripper_width: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.datasets = [
            BaseLiberoDataset(
                *args, 
                dataset_name=dataset_name,
                act_step=act_step,
                small_size=small_size,
                gripper_width=gripper_width,
                **kwargs,
                
            ) for dataset_name in dataset_names
        ]
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.traj_cons = traj_cons
        self.act_step = act_step
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        self.length_each_dataset = [len(dataset) for dataset in self.datasets]
        self.accumulated_length_each_dataset = list(accumulate(self.length_each_dataset))

    def register_image_preprocess_hook(self, func):
        self.image_preprocess = func

    def __len__(self):
        return self.accumulated_length_each_dataset[-1]
    
    def __getitem__(self, idx):
        dataset_id = bisect.bisect_right(self.accumulated_length_each_dataset, idx)
        if dataset_id - 1 >= 0:
            local_idx = idx - self.accumulated_length_each_dataset[dataset_id - 1]
        else:
            local_idx = idx

        return self.datasets[dataset_id].__getitem__(local_idx)

    def collator(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        episode_id = [s["episode_id"] for s in sample]
        text_tensors = self.text_fn(stacked_language)

        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        
        robot_obs = torch.zeros(1)

        if self.act_step != 1:
        
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]

            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)

            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]

        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs 

def get_libero_pretrain_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_names = ["libero_90_converted"]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)

    libero_dataset = DiskLiberoDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        dataset_info='libero_90_converted',
        gripper_width=args.gripper_width,
        load_libero_file=args.load_libero_file,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(libero_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        libero_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        libero_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=libero_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=libero_dataset)

def get_libero_finetune_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_names = ["libero_10_converted"]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)

    libero_dataset = DiskLiberoDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        dataset_info='libero_10_converted',
        gripper_width=args.gripper_width,
        load_libero_file=args.load_libero_file,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(libero_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        libero_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        libero_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=libero_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=libero_dataset)


def get_real_finetune_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_names = [args.real_dataset_names]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    real_dataset = DiskRealDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        dataset_info=args.real_dataset_names,
        gripper_width=args.gripper_width,
        load_real_file="npz",
        use_aug_data=args.use_aug_data,
        max_rel_pos=args.max_rel_pos,
        max_rel_orn=args.max_rel_orn,
        magic_scaling_factor_pos=args.magic_scaling_factor_pos,
        magic_scaling_factor_orn=args.magic_scaling_factor_orn,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(real_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        real_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=real_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=real_dataset)

class BaseOXEDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        image_primary_size=200,
        image_wrist_size=84,
        obs_space: DictConfig = obs_config,
        proprio_state: DictConfig = prop_state,
        transforms: Dict = {},
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        text_aug=False,
        dif_ws=False,
        act_step: int = 1,
        key: str = "lang",
        language_mode: str = "language_instruction",
        primary_mode: str = "image_primary",
        small_size: int = 0,
        data_in_ceph: bool = False, 
        max_rel_pos: float = 0.02,
        max_rel_orn: float = 0.05,
        magic_scaling_factor_pos: float = 1.0,
        magic_scaling_factor_orn: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__()
        
        self.dataset_name = dataset_name 
        self.dataset_info = dataset_name
        self.root_dir = root_dir 
        self.dataset_path = f'{root_dir}/{dataset_name}' 
        self.conf_path = '~/petreloss.conf'
        self.client = Client(self.conf_path)
        self.image_primary_size = image_primary_size
        self.image_wrist_size = image_wrist_size
        self.image_preprocess = None
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.window_size = window_size
        self.language_mode = language_mode
        self.primary_mode = primary_mode
        self.small_size = small_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            raise NotImplementedError
        assert self.max_window_size == self.min_window_size
        self.aux_lang_loss_window = aux_lang_loss_window
        self.text_aug = text_aug
        self.act_step = act_step
        self.data_in_ceph = data_in_ceph
        self.max_rel_pos = max_rel_pos
        self.max_rel_orn = max_rel_orn
        self.magic_scaling_factor_pos = magic_scaling_factor_pos
        self.magic_scaling_factor_orn = magic_scaling_factor_orn
        logger.info(f"loading dataset at {root_dir}/{dataset_name}")
        logger.info("finished loading dataset")
        if self.data_in_ceph:
            assert self.client.isdir(self.dataset_path)
        else:
            os.path.exists(self.dataset_path)
        assert os.path.exists(f"data_info/{self.dataset_info}.json")
        with open(f"data_info/{self.dataset_info}.json", 'r') as f: #TODO
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list[1:]]
            self.transform_wrist_image = self.episode_info_list[0]["wrist_image"]
            self.num_step_per_episode = [f[1] - self.max_window_size for f in self.episode_info_list[1:]]
            self.num_episode = len(self.episode_list)
        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))
        
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        
        return padded

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        
        return seq

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} != max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx

        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)

        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list
        
        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
        if episode_id - 1 >= 0:
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = idx
        num_step_per_episode = self.num_step_per_episode[episode_id]
        end_id = min(start_id + window_size, num_step_per_episode)
        episode_id = self.episode_list[episode_id]
        episodes = []
        for step_id in range(start_id, end_id):
            data_dict = {}
            try:
                step_id = str(step_id).zfill(4)
                other_path = f"{self.dataset_path}/{episode_id}/steps/{step_id}/other.h5"
                if self.data_in_ceph:
                    other_bytes = self.client.get(other_path, enable_cache=True)
                    other_h5_file = h5py.File(io.BytesIO(other_bytes))
                else:
                    other_h5_file = h5py.File(other_path)
            except:
                print("episode_id :", episode_id)
                print("step_id :", step_id)
                print("num_step_per_episode :", num_step_per_episode)
                print("other_path", f"{self.dataset_path}/{episode_id}/steps/{step_id}/other.h5")
            data_dict["rgb_static"] = self.load_primary_rgb(episode_id, step_id, self.primary_mode)
            data_dict["rgb_gripper"] = self.load_wrist_rgb(episode_id, step_id)
            data_dict["rel_actions"] = self.load_action(other_h5_file)
            try:
                data_dict["robot_obs"] = self.load_robot_obs(other_h5_file)
            except:
                print(f"{self.dataset_path}/{episode_id}/steps/{step_id}/other.h5")
                raise NotImplementedError
            data_dict["scene_obs"] = self.load_scene_obs(episode_id, step_id)
            episodes.append(data_dict)
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode["language"] = self.load_language_instruction(other_h5_file, self.language_mode)
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        info["use_for_aux_lang_loss"] = False
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  
        seq_dict["droid_episode_id"] = episode_id

        return seq_dict

    def load_primary_rgb(self, episode_id, step_id, primary_mode="image_primary"):
        image_primary = f'{self.dataset_path}/{episode_id}/steps/{step_id}/{primary_mode}.jpg'
        if self.data_in_ceph:
            image_primary = self.client.get(image_primary, enable_cache=True)
            image_primary = io.BytesIO(image_primary)
        image_primary = np.array(Image.open(image_primary).convert("RGB")) 
        
        return image_primary.astype(np.uint8)

    def load_wrist_rgb(self, episode_id, step_id):
        image_wrist = f'{self.dataset_path}/{episode_id}/steps/{step_id}/image_wrist.jpg'
        if self.data_in_ceph:
            image_wrist = self.client.get(image_wrist, enable_cache=True)
            image_wrist = io.BytesIO(image_wrist)
        image_wrist = np.array(Image.open(image_wrist).convert("RGB")) 
        if self.transform_wrist_image ==  "Flip vertically & horizontally":
            image_wrist = np.flip(image_wrist, axis=1) 
            image_wrist = np.flip(image_wrist, axis=0)
        
        return image_wrist.astype(np.uint8)

    def load_language_instruction(self, other_h5_file, language_mode="language_instruction"):
        language_instruction = other_h5_file[language_mode][()].decode('utf-8')
        
        return language_instruction
        
    def load_action(self, other_h5_file, max_rel_pos=0.02, max_rel_orn=0.05, magic_scaling_factor_pos=1.0, magic_scaling_factor_orn=1.0):
        action = other_h5_file["action_delta_wrist_pose"][()]
        if self.dataset_name in [
            f"furniture_bench_dataset_converted_externally_to_rlds",
            f"berkeley_autolab_ur5",
            f"berkeley_fanuc_manipulation",
            ]:
            action[:3] /= (self.max_rel_pos * 10.0)
            action[3:6] /= (self.max_rel_orn * 10.0)
        else:
            action[:3] /= (self.max_rel_pos * self.magic_scaling_factor_pos)
            action[3:6] /= (self.max_rel_orn * self.magic_scaling_factor_orn)

        return action

    def load_robot_obs(self, other_h5_file):
        robot_obs = np.zeros(self.proprio_state.n_state_obs)
        robot_obs[:6] = other_h5_file['observation']['gripper_pose6d'][()]
        robot_obs[-1] = other_h5_file['observation']['gripper_open_state'][()][0]
        if self.dataset_name in ["berkeley_autolab_ur5", "berkeley_fanuc_manipulation", "jaco_play"]:
            pass 
        else:
            robot_obs[7:14] = other_h5_file['observation']['joint_position'][()]
        
        return robot_obs

    def load_scene_obs(self, episode_id, step_id):
        scene_obs = np.zeros(self.proprio_state.n_scene_obs)
        
        return scene_obs

    def __len__(self):
        if self.small_size:
            return self.small_size
        else:
            return self.length

class DistOXEDataset(Dataset):
    def __init__(
        self, 
        image_fn: Callable,
        text_fn: Callable,
        dataset_names: List[str],
        *args: Any,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        traj_cons: bool = False,
        act_step : int = 1,
        small_size: int = 0, 
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.datasets = [
            BaseOXEDataset(
                *args, 
                dataset_name=dataset_name,
                act_step=act_step,
                small_size=small_size,
                **kwargs,
                
            ) for dataset_name in dataset_names
        ]
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.traj_cons = traj_cons
        self.act_step = act_step
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        self.length_each_dataset = [len(dataset) for dataset in self.datasets]
        self.accumulated_length_each_dataset = list(accumulate(self.length_each_dataset))

    def register_image_preprocess_hook(self, func):
        self.image_preprocess = func

    def __len__(self):
        return self.accumulated_length_each_dataset[-1]
    
    def __getitem__(self, idx):
        dataset_id = bisect.bisect_right(self.accumulated_length_each_dataset, idx)
        if dataset_id - 1 >= 0:
            local_idx = idx - self.accumulated_length_each_dataset[dataset_id - 1]
        else:
            local_idx = idx

        return self.datasets[dataset_id].__getitem__(local_idx)


    def collator(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        droid_episode_id = [s["droid_episode_id"] for s in sample]
        text_tensors = self.text_fn(stacked_language)
        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        robot_obs = torch.zeros(1)
        if self.act_step != 1:
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]

        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs 

def get_oxe_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_names = [
        ## other robots
        "berkeley_autolab_ur5",
        "jaco_play",
        ## franka
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "viola",
        "stanford_hydra_dataset_converted_externally_to_rlds",
        "berkeley_fanuc_manipulation",
        "austin_buds_dataset_converted_externally_to_rlds",
        "utaustin_mutex",
        "taco_play",
        "austin_sailor_dataset_converted_externally_to_rlds",
        "austin_sirius_dataset_converted_externally_to_rlds",
        "furniture_bench_dataset_converted_externally_to_rlds",
    ]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    oxe_dataset = DistOXEDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        data_in_ceph=args.data_in_ceph,
        max_rel_pos=args.max_rel_pos,
        max_rel_orn=args.max_rel_orn,
        magic_scaling_factor_pos=args.magic_scaling_factor_pos,
        magic_scaling_factor_orn=args.magic_scaling_factor_orn,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(oxe_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        oxe_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        oxe_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=oxe_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=oxe_dataset)




def depth_image_fn(depth_images, normalize=True, max_depth=10.0, to_tensor=True):
    
    np_depth = np.stack([np.array(img, dtype=np.float32) for img in depth_images])
    if len(np_depth.shape) != 3:
        raise ValueError("Depth images should have shape (N, H, W)")
    eps = 1e-6
    # np_depth_norm = (np_depth - np_depth.min(axis=(1,2), keepdims=True)) / (np_depth.max(axis=(1,2), keepdims=True) - np_depth.min(axis=(1,2), keepdims=True) + eps)
    np_depth_norm = np_depth
    depth_tensor = torch.from_numpy(np_depth_norm).unsqueeze(1)  # (N, 1, H, W)

    resize = T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST)
    depth_tensor_resized = torch.stack([resize(d) for d in depth_tensor])  
    # for img in np_depth])
    if to_tensor:
        # depth_tensors = torch.from_numpy(resized_depth).unsqueeze(1)# 调整维度顺序为 (N, C, H, W)
        return depth_tensor_resized
    else:
        return np_depth


if __name__ == "__main__":
    from arguments_utils import get_args_and_cfg
    from tqdm import tqdm
    args, _ = get_args_and_cfg()
    device='cuda'
    model, image_processor = clip.load("ViT-B/32", device=device)
    calvin_dataset = get_libero_pretrain_dataset(args, image_processor, clip, epoch=0)
    calvin_dataset.set_epoch(epoch=0)
    calvin_loader = calvin_dataset.dataloader
    num_batches_per_epoch = calvin_loader.num_batches
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=num_batches_per_epoch,
    )
    mv_avg_loss = []
    for num_steps, batch in t:
        if num_steps > 0:
            torch.cuda.synchronize()
            t2 = time.time()
            print("t2 - t1", t2 - t1)
        torch.cuda.synchronize()
        t1 = time.time()
