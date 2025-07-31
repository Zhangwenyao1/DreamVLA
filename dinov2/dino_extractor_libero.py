import os
import argparse
import math
from pathlib import Path
from typing import Sequence 
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from natsort import natsorted
import glob
from tqdm import tqdm
import time
import numpy as np
import torch
import torchvision.transforms as tvt
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imageio
from dinov2.hub.backbones import dinov2_vitb14
import dist_utils
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> tvt.Normalize:
    return tvt.Normalize(mean=mean, std=std)

class LIBERODataset(Dataset):
    def __init__(self, data_root, seq_name, img_transform=None, except_lang=False):
        super().__init__()
        self.data_root = data_root

        all_files = []
        epi_id = seq_name
        seq_files = natsorted(os.listdir(os.path.join(data_root, epi_id, 'steps'))) 
        for frame_id in seq_files:
            all_files.append(os.path.join(data_root, epi_id, 'steps', frame_id, 'image_primary.jpg'))
        for frame_id in seq_files:
            all_files.append(os.path.join(data_root, epi_id, 'steps', frame_id, 'image_wrist.jpg'))
        self.all_files = all_files
            
        self.transform = img_transform
            
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_name = self.all_files[idx]
        
        img = Image.open(file_name).resize((224, 224), resample=Image.Resampling.BICUBIC)
        # 不知道为啥libero的image_primary是上下颠倒的
        if 'image_primary' in file_name:
            img = np.array(img)[::-1].copy()
        else:
            img = np.array(img)
        img = self.transform(img)
        return {'img': img, 'file_path': file_name}


def save_result(file_name, save_path, data):
    # file_name = batch['file_path'][i]
    epi_id = file_name.split('/')[-4]
    frame_id = file_name.split('/')[-2]
    _save_path = os.path.join(save_path, epi_id, 'steps', frame_id)
    os.makedirs(_save_path, exist_ok=True)
    data.to(torch.bfloat16).cpu()
    # torch.save(data.to(torch.bfloat16).cpu(), os.path.join(_save_path, file_name.split('/')[-1].split('.')[0]+'.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        default="libero_goal_converted/dinov2_feats",
        type=str
    )
    parser.add_argument(
        "--data_root",
        default="libero_goal_converted/episodes", #replace with your data path
        type=str
    )
    parser.add_argument(
        "--split",
        default="training",
        type=str
    )
    parser.add_argument(
        "--checkpoint",
        default="dinov2/ckpts/dinov2_vitb14_pretrain.pth",
        # default=None,
        help="dino model parameters",
    )
    parser.add_argument(
        "--start_idx",
        default=0,
        type=int
    )
    parser.add_argument(
        "--end_idx",
        default=100000000,
        type=int
    )
    parser.add_argument(
        "--seq_to_process",
        type=str,
        default=None
    )
    parser.add_argument(
        "--override_exist_files",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )
    parser.add_argument(
        '--except_lang',
        action='store_true',
        default=False
    )

    args = parser.parse_args()

    if dist_utils.is_dist():
        dist_utils.ddp_setup()
        rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
    else:
        rank = 0
        world_size = 1
    print('rank: ', rank, 'world_size: ', world_size)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    data_root = args.data_root

    # model
    model = dinov2_vitb14(pretrained=False)
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt, strict=True)
    model = model.to('cuda')
    img_transform = tvt.Compose([
        tvt.ToTensor(),
        make_normalize_transform()
    ])

    # dataset
    total_seq = os.listdir(data_root)
    total_seq = natsorted([item for item in total_seq if (not item.endswith('.h5'))])
    chunk_size = math.ceil(len(total_seq) / world_size)
    start = rank * chunk_size
    end = min(start + chunk_size, len(total_seq))
    sub_seq = total_seq[start:end]
    
    for seq_name in tqdm(sub_seq):
        
        dataset = LIBERODataset(data_root, seq_name, img_transform=img_transform, except_lang=args.except_lang)
        sampler = None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, sampler=sampler, pin_memory=True, persistent_workers=True)

        data_time_end = time.time()
        results = []
        for batch in dataloader:
            data_time = time.time() - data_time_end
            # check existing
            if 0:
                _save_path = os.path.join(args.save_path, seq_name+'.pt')
                if os.path.exists(_save_path):
                    try:
                        torch.load(_save_path)
                        continue
                    except:
                        pass
            
            imgs = batch['img'].to('cuda')
            
            with torch.no_grad():
                out = model(imgs, is_training=True)

            save_time_start = time.time()
            
            for i in range(len(batch['file_path'])):
                file_name = batch['file_path'][i]
                epi_id = file_name.split('/')[-4]
                frame_id = file_name.split('/')[-2]
                _save_path = os.path.join(args.save_path, epi_id, 'steps', frame_id)
                os.makedirs(_save_path, exist_ok=True)
                #torch.save(out['x_norm_patchtokens'][i].to(torch.bfloat16).cpu(), os.path.join(_save_path, file_name.split('/')[-1].split('.')[0]+'.pt'))
                np.save(os.path.join(_save_path, file_name.split('/')[-1].split('.')[0]+'.npy'), out['x_norm_patchtokens'][i].to(torch.float32).cpu().numpy())

            
            # results.append(out['x_norm_patchtokens'])       
            save_time = time.time() - save_time_start
            data_time_end = time.time()
        # results = torch.cat(results)
        # torch.save(results.to(torch.bfloat16).cpu(), os.path.join(args.save_path, seq_name+'.pt'))
            
    if dist_utils.is_dist():
        dist_utils.ddp_cleanup()
