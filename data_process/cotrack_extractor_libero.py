import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import cv2
import imageio
import dist_utils
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor

def visualize_optical_flow(flow: np.ndarray, convert_to_bgr=False) -> np.ndarray:
    h, w = flow.shape[:2]
    flow_map = np.zeros((h, w, 3), dtype=np.uint8)

    dx = flow[..., 0]
    dy = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)       
    hsv[..., 1] = 255                                
    hsv[..., 2] = np.clip((magnitude * 8), 0, 255).astype(np.uint8)  

    flow_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR if convert_to_bgr else cv2.COLOR_HSV2RGB)
    return flow_map

def load_episode(j, image_key='rgb_static'):
    ep = np.load(os.path.join(data_root, f'episode_{j:0{n_digits}d}.npz'))
    img = ep[image_key]
    img = Image.fromarray(img).resize((224, 224))
    return img

def save_track_label(j, trk, vis):
    # import pdb; pdb.set_trace()
    file_path = os.path.join(args.save_path, f'{j}.npz')
    if os.path.exists(file_path):
        print(f"File {file_path} already exists, skipping...")
        return
    with open(os.path.join(args.save_path, f'{j}.npz'), 'wb') as f:
        np.savez_compressed(f, 
                            tracks=trk, 
                            visibility=vis)
    
def get_points_on_a_grid(patch_size, image_size, device):

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    H, W = image_size
    ph, pw = patch_size

    assert H % ph == 0 and W % pw == 0, "The patch size must divide the image dimensions"

    y_centers = np.arange(ph // 2, H, ph)
    x_centers = np.arange(pw // 2, W, pw)

    xv, yv = np.meshgrid(x_centers, y_centers)
    centers = np.stack([xv, yv], axis=-1).reshape(-1, 2)
    return torch.from_numpy(centers).to(device)

class LIBERODataset(Dataset):
    def __init__(self, data_root, frame_gap, grid_query_frame):
        super().__init__()
        self.data_root = data_root
        self.frame_gap = frame_gap
        self.grid_query_frame = grid_query_frame

        all_episodes = natsorted(os.listdir(data_root))
        all_seqs = []
        for epi_id in all_episodes:
            all_files_primary = []
            all_files_wrist = []
            seq_files = natsorted(os.listdir(os.path.join(data_root, epi_id, 'steps'))) 
            for frame_id in seq_files:
                all_files_primary.append(os.path.join(data_root, epi_id, 'steps', frame_id, 'image_primary.jpg'))
            for frame_id in seq_files:
                all_files_wrist.append(os.path.join(data_root, epi_id, 'steps', frame_id, 'image_wrist.jpg'))
            all_seqs.append(all_files_primary)
            all_seqs.append(all_files_wrist)
            
        self.all_episodes = all_seqs

    def __len__(self):
        return len(self.all_episodes)

    def __getitem__(self, idx):
        all_files = self.all_episodes[idx]
        epi_id = all_files[0].split('/')[-4]
        img_key = all_files[0].split('/')[-1].split('.')[0]
        video = []
        for file_name in all_files:
            img = Image.open(file_name).resize((224, 224), resample=Image.Resampling.BICUBIC)
            # 不知道为啥libero的image_primary是上下颠倒的
            if 'image_primary' in file_name:
                img = np.array(img)[::-1].copy()
            else:
                img = np.array(img)
            video.append(img)
            
        video = np.stack(video)
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        if video.shape[0] < self.frame_gap+1:
            video = torch.zeros((0, 2, *video.shape[1:]))
        else:
            video = video.unfold(0, self.frame_gap+1, 1)
            video = video[..., [0, -1]]
            # video = video.unfold(0, 2, 1)
            video = video.permute(0, 4, 1, 2, 3).contiguous() # [N, 3, 224, 224, 2] -> [N, 2, 3, 224, 224]

        queries = torch.cat(
            [torch.ones_like(grid_pts[:, :, :1]) * self.grid_query_frame, grid_pts],
            dim=2,
        ).repeat(video.shape[0], 1, 1)
        ret = dict(
            video=video,
            queries=queries,
            epi_id=epi_id,
            img_key=img_key
        )
        return ret
    
    @staticmethod
    def collect_fn(batches):
        assert len(batches) == 1
        return batches[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        default="libero_object_converted/cotracker_traj",
        type=str
    )
    parser.add_argument(
        "--data_root",
        default="libero_object_converted/episodes", # replace with your data path
        type=str
    )
    parser.add_argument(
        "--split",
        default="training",
        type=str
    )
    parser.add_argument(
        "--image_key",
        default='rgb_gripper',
        type=str
    )
    parser.add_argument(
        "--checkpoint",
        default="./co-tracker/checkpoints/scaled_offline.pth",
        # default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--patch_size", type=int, default=8)
    #parser.add_argument("--grid_size", type=int, default=0, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
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
        "--frame_gap",
        default=3,
        type=int
    )
    parser.add_argument(
        "--except_lang",
        action='store_true',
        default=False
    )

    args = parser.parse_args()

    if dist_utils.is_dist():
        dist_utils.ddp_setup()
        rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
    else:
        rank, world_size = 0, 1
    
    print(f'Rank: {rank}, World size: {world_size}')

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    data_root = args.data_root

    # dataset
    
    dataset = LIBERODataset(data_root, args.frame_gap, args.grid_query_frame)
    if dist_utils.is_dist():
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collect_fn, sampler=sampler)

    # model
    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
        else:
            if args.offline:
                window_len = 60
            else:
                window_len = 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    model = model.to('cuda')

    if args.image_key == 'rgb_static':
        H, W = 224, 224
    else:
        H, W = 224, 224
    grid_pts = get_points_on_a_grid(
        args.patch_size, [H, W], device='cpu'
    ).float().unsqueeze(0)

    args.grid_size = H // args.patch_size

    n_digits = 7
    loader = ThreadPoolExecutor(max_workers=32)
    saver = ThreadPoolExecutor(max_workers=32)
    all_exists = set(entry.name for entry in os.scandir(args.save_path))
    total_len = len(dataset)
    print(f"Total len: {total_len}")
    # process_seqs = list(range(args.start_idx, min(args.end_idx+1, total_len)))
    # print(f"Process len: {len(process_seqs)}")

    if args.seq_to_process is not None:
        with open(args.seq_to_process, 'r') as f:
            seq_to_process = f.readlines()
        seq_to_process = [item.strip() for item in seq_to_process]
    else:
        seq_to_process = []

    for batch in tqdm(dataloader):
        if batch is None:
            continue
        video = batch['video'].cuda()
        queries = batch['queries'].cuda()
        epi_id = batch['epi_id']
        img_key = batch['img_key']
        pred_tracks = []
        pred_visibility = []
        batch_size = 32
        total_size = video.shape[0]
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            pred_tracks_batch, pred_visibility_batch = model(
                video[start:end],
                queries=queries[start:end],
                grid_size=args.grid_size,
                #grid_query_frame=args.grid_query_frame,
                backward_tracking=args.backward_tracking,
                # segm_mask=segm_mask
            )
            pred_tracks.append(pred_tracks_batch)
            pred_visibility.append(pred_visibility_batch)
        
        pred_tracks = torch.cat(pred_tracks)
        pred_visibility = torch.cat(pred_visibility)

        pred_tracks_delta = (pred_tracks[:, 1:2, :, :] - pred_tracks[:, 0:1, :, :]).squeeze(1)

        tracks = dict(tracks=pred_tracks_delta.cpu().numpy(), visibility=pred_visibility[:, 1, :].cpu().numpy())
        tracks['tracks'] = np.concatenate([tracks['tracks'], np.zeros([args.frame_gap, 784, 2],dtype=np.float32)], axis=0)
        tracks['visibility'] = np.concatenate([tracks['visibility'], np.zeros([args.frame_gap, 784],dtype=np.float32)], axis=0)
        # with open(os.path.join(args.save_path, f'{epi_id}_{img_key}.npz'), 'wb') as f:
        #     np.savez_compressed(f, tracks=tracks['tracks'], visibility=tracks['visibility'])
        
        for j in range(tracks["tracks"].shape[0]):
            trk = tracks['tracks'][j]
            vis = tracks['visibility'][j]
            _save_path = os.path.join(args.save_path, epi_id, 'steps', f'{j:04d}')
            os.makedirs(_save_path, exist_ok=True)
            with open(os.path.join(_save_path, f'{img_key}.npz'), 'wb') as f:
                np.savez_compressed(f, tracks=tracks['tracks'][j], visibility=tracks['visibility'][j])

        vis = False
        if vis:
            # B, T, N, D = pred_tracks.shape
            # for j in range(B):

            # video = video.view(1, -1, 3, 224, 224)
            # video = video[:, 1::2, :, :, :]
            # pred_tracks = pred_tracks.view(1, -1, pred_tracks.shape[-2], 2)
            # init_grid = pred_tracks[:, 0:1, :, :]
            # pred_tracks = pred_tracks[:, 1::2, :, :]
            # pred_visibility = pred_visibility.view(1, -1, pred_visibility.shape[-1])
            # pred_visibility = pred_visibility[:, 1::2, :]

            # # visualize flow map
            # pred_tracks_delta = (pred_tracks - init_grid).view(1, -1, args.grid_size, args.grid_size, 2)
            # flow_maps = []
            # for j in range(pred_tracks_delta.shape[1]):
            #     flow_map = visualize_optical_flow(pred_tracks_delta[0, j, :, :, :].cpu().numpy())
            #     flow_maps.append(flow_map)
            # writer = imageio.get_writer(os.path.join("./saved_videos", f'{i}_{lang_description}_flow_map.mp4'))
            # for flow_map in flow_maps:
            #     writer.append_data(flow_map)
            # writer.close()


            # save a video with predicted tracks
            vis = Visualizer(save_dir="./saved_videos", pad_value=10, linewidth=1, show_first_frame=1)

            # choice moving points
            #assert pred_tracks.shape[0] == 1 # batch size == 1
            #motion_mask = (pred_tracks[:, 1:, :, :] - pred_tracks[:, :-1, :, :]).abs().sum(-1).sum(1) > 20
            #pred_tracks = pred_tracks[:, :, motion_mask.squeeze(), :]
            #pred_visibility = pred_visibility[:, :, motion_mask.squeeze()]
            vis.visualize(
                video,
                pred_tracks,
                pred_visibility,
                query_frame=0 if args.backward_tracking else args.grid_query_frame,
                filename=f'{i}_{lang_description}'
            )
        
    if dist_utils.is_dist():
        dist_utils.ddp_cleanup()
