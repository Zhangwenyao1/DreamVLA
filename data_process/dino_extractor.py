import os
import argparse
from pathlib import Path
from typing import Sequence 
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from natsort import natsorted
import glob
from tqdm import tqdm
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

def load_episode(data_root, j, image_key='rgb_static'):
    n_digits = 7
    ep = np.load(os.path.join(data_root, f'episode_{j:0{n_digits}d}.npz'))
    img = ep[image_key]
    img = Image.fromarray(img).resize((224, 224))
    return img

class CalvinDataset(Dataset):
    def __init__(self, data_root, image_key='rgb_static', img_transform=None, except_lang=False):
        super().__init__()
        self.data_root = data_root
        self.image_key = image_key

        if not except_lang:
            lang_file = os.path.join(data_root, 'lang_annotations/auto_lang_ann.npy')
            self.lang = np.load(lang_file, allow_pickle=True).item()
        else:
            self.lang = {'info':{'indx':None}}
            ep_start_end_ids = np.load(Path(data_root) / "except_lang_idx" / "except_lang_idx.npy").tolist()
            self.lang['info']['indx'] = ep_start_end_ids

        # self.all_files = glob.glob(os.path.join(data_root, 'episode_*.npz')) # 会比真正用到的要多很多
        _temp = []
        for item in self.lang['info']['indx']:
            _temp.extend(list(range(item[0], item[1]+1)))
        self.all_indx = sorted(list(set(_temp)))
            
        self.transform = img_transform
            
    def __len__(self):
        return len(self.all_indx)

    def __getitem__(self, idx):
        # file_name = os.path.basename(self.all_files[j])
        # j = int(file_name.split('.')[0].split('_')[-1])
        file_name = self.all_indx[idx]
        img = load_episode(self.data_root, file_name, self.image_key)
        img = np.array(img)
        img = self.transform(img)
        return {'img': img, 'idx': file_name}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        default="calvin_dino/task_ABC_D",
        type=str
    )
    parser.add_argument(
        "--data_root",
        default="/mnt/afs/chenxuchuan/datasets/calvin/task_ABC_D",
        type=str
    )
    parser.add_argument(
        "--split",
        default="training",
        type=str
    )
    parser.add_argument(
        "--image_key",
        default='rgb_static',
        type=str
    )
    parser.add_argument(
        "--checkpoint",
        default="./ckpts/sam_vit_b_01ec64.pth",
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
    args.save_path = os.path.join(args.save_path, args.image_key, args.split)

    if dist_utils.is_dist():
        dist_utils.ddp_setup()
        rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
    else:
        rank = 0
        world_size = 1
    print('rank: ', rank, 'world_size: ', world_size)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    data_root = os.path.join(args.data_root, args.split)

    # model
    model = dinov2_vitb14(pretrained=False)
    ckpt = torch.load('ckpts/dinov2_vitb14_pretrain.pth')
    model.load_state_dict(ckpt, strict=True)
    model = model.to('cuda')
    img_transform = tvt.Compose([
        tvt.ToTensor(),
        make_normalize_transform()
    ])

    # dataset
    dataset = CalvinDataset(data_root, args.image_key, img_transform=img_transform, except_lang=args.except_lang)
    if dist_utils.is_dist():
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, sampler=sampler)

    print(f"[Rank {rank}] Total batches: {len(dataloader)}")

    for batch in tqdm(dataloader, desc=f"[Rank {rank}]"):
        imgs = batch['img'].to('cuda')
        
        with torch.no_grad():
            out = model(imgs, is_training=True)

        for i in range(len(batch['idx'])):
            torch.save(out['x_norm_patchtokens'][i].to(torch.bfloat16).cpu(), os.path.join(args.save_path, f'{batch["idx"][i].item()}.pt'))

    if dist_utils.is_dist():
        dist_utils.ddp_cleanup()
