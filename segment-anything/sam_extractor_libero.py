import os
import argparse
from typing import Sequence 
import math
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
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import dist_utils

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


class LIBERODataset(Dataset):
    def __init__(self, data_root, seq_name, input_size):
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
            
        self.transform = ResizeLongestSide(input_size)
            
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_name = self.all_files[idx]
        img = Image.open(file_name)
        # 不知道为啥libero的image_primary是上下颠倒的
        if 'image_primary' in file_name:
            img = np.array(img)[::-1].copy()
        else:
            img = np.array(img)
        img = self.transform.apply_image(img)
        input_image_torch = torch.as_tensor(img)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        return {'img': input_image_torch, 'file_path': file_name}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        default="libero_object_converted/sam_feats",
        type=str
    )
    parser.add_argument(
        "--data_root",
        default="libero_object_converted/episodes", #replace with your data path
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
        default="./segment-anything/ckpts/sam_vit_b_01ec64.pth",
        # default=None,
        help="sam model parameters",
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
        default=16
    )

    args = parser.parse_args()

    if dist_utils.is_dist():
        dist_utils.ddp_setup()
        rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
    else:
        rank, world_size = 0, 1
    print('rank: ', rank, 'world_size: ', world_size)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    data_root = args.data_root

    # model
    model_type = '_'.join(os.path.basename(args.checkpoint).split('_')[1:3])
    sam = sam_model_registry[model_type](checkpoint=args.checkpoint)
    
    sam.to('cuda')
    
    mask_generator = SamAutomaticMaskGenerator(sam)

    # dataset
    total_seq = os.listdir(data_root)
    total_seq = natsorted([item for item in total_seq if (not item.endswith('.h5'))])
    chunk_size = math.ceil(len(total_seq) / world_size)
    start = rank * chunk_size
    end = min(start + chunk_size, len(total_seq))
    sub_seq = total_seq[start:end]
    
    for seq_name in tqdm(sub_seq):
        
        dataset = LIBERODataset(data_root, seq_name, sam.image_encoder.img_size)
        sampler = None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, sampler=sampler, pin_memory=True, persistent_workers=True)

        total_len = len(dataloader)
        results = []
        for batch in dataloader:
            imgs = batch['img'].to('cuda')
            output_mask = False
            with torch.no_grad():
                if output_mask:
                    raise NotImplementedError
                    # masks = mask_generator.generate(img)
                    # if True:
                    #     plt.figure(figsize=(20,20))
                    #     plt.imshow(img)
                    #     show_anns(masks)
                    #     plt.axis('off')
                    #     plt.savefig('demo.png')
                else:
                    # mask_generator.predictor.set_image(imgs)
                    imgs = sam.preprocess(imgs)
                    features = sam.image_encoder(imgs) # [B, C, H, W]
                    
                    # [64, 64] -> [16, 16]
                    features = torch.nn.functional.avg_pool2d(features, kernel_size=4, stride=4, padding=0)
                    # features = torch.nn.functional.interpolate(features, size=(16,16), mode='bilinear', align_corners=False)

                    features = features.flatten(start_dim=-2)

        #     results.append(features.to(torch.bfloat16).cpu())  
            for i in range(len(batch['file_path'])):
                file_name = batch['file_path'][i]
                epi_id = file_name.split('/')[-4]
                frame_id = file_name.split('/')[-2]
                _save_path = os.path.join(args.save_path, epi_id, 'steps', frame_id)
                os.makedirs(_save_path, exist_ok=True)
                # torch.save(features[i].to(torch.bfloat16).cpu(), os.path.join(args.save_path, epi_id, 'steps', frame_id, file_name.split('/')[-1].split('.')[0]+'.pt'))
                np.save(os.path.join(_save_path, file_name.split('/')[-1].split('.')[0]+'.npy'), features[i].to(torch.float32).cpu().numpy())

        # results = torch.cat(results)
        # torch.save(results, os.path.join(args.save_path, seq_name+'.pt'))
        
    if dist_utils.is_dist():
        dist_utils.ddp_cleanup()
