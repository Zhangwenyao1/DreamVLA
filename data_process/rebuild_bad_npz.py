#!/usr/bin/env python3
import os
import re
import argparse
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from cotracker.predictor import CoTrackerPredictor


def atomic_save_npz(final_path: str, **arrays):
    d = os.path.dirname(final_path)
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=d, delete=False) as tf:
        tmp = tf.name
        np.savez_compressed(tf, **arrays)
        tf.flush()
        os.fsync(tf.fileno())
    os.replace(tmp, final_path)


def load_image_from_episode(ep_path: str, image_key: str, resize_hw: Tuple[int, int]) -> np.ndarray:
    # 安全读取 + 只读 mmap
    with np.load(ep_path, mmap_mode="r") as ep:
        img = ep[image_key]
    img = Image.fromarray(img).resize(resize_hw)
    return np.asarray(img)


def build_grid_queries(patch_size: int, H: int, W: int, grid_query_frame: int) -> torch.Tensor:
    assert H % patch_size == 0 and W % patch_size == 0
    y_centers = np.arange(patch_size // 2, H, patch_size)
    x_centers = np.arange(patch_size // 2, W, patch_size)
    xv, yv = np.meshgrid(x_centers, y_centers)  # [h,w]
    centers = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # [N,2] (x,y)
    frame_col = np.full((centers.shape[0], 1), grid_query_frame, dtype=np.float32)
    q = np.concatenate([frame_col, centers.astype(np.float32)], axis=1)  # [N,3]
    return torch.from_numpy(q).unsqueeze(0)  # [1,N,3]


def parse_bad_path(p: str):
    """
    从坏文件路径中解析 image_key, split, j：
    .../<image_key>/<split>/<j>.npz
    """
    path = Path(p)
    j = int(path.stem)
    split = path.parent.name
    image_key = path.parent.parent.name
    return image_key, split, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bad_list", type=str, required=True,
                    help="包含坏文件完整路径的 txt（每行一个路径）")
    ap.add_argument("--data_root", type=str, required=True,
                    help="CALVIN 原始数据根目录（其下有 training/validation/.../episode_*.npz）")
    ap.add_argument("--checkpoint", type=str, default="./checkpoints/scaled_offline.pth")
    ap.add_argument("--offline", action="store_true", default=True)
    ap.add_argument("--use_v2_model", action="store_true")
    ap.add_argument("--frame_gap", type=int, default=3)
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--grid_query_frame", type=int, default=0)
    ap.add_argument("--resize_h", type=int, default=224)
    ap.add_argument("--resize_w", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    # 加载 CoTracker（与原来一致：offline=>window_len=60，否则16）
    if args.use_v2_model:
        model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=True)
    else:
        window_len = 60 if args.offline else 16
        model = CoTrackerPredictor(
            checkpoint=args.checkpoint, v2=False, offline=args.offline, window_len=window_len
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    H, W = args.resize_h, args.resize_w
    grid = build_grid_queries(args.patch_size, H, W, args.grid_query_frame).to(device)  # [1,N,3]
    grid_size = H // args.patch_size  # = W // patch_size

    # 读取坏文件列表
    bad_paths = []
    with open(args.bad_list, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 行里可能有前缀标记（如 [BAD-ZIP] <path>），只提取路径
            m = re.search(r"(/.*\.npz)$", line)
            if m:
                bad_paths.append(m.group(1))
            elif line.endswith(".npz"):
                bad_paths.append(line)

    print(f"[INFO] Found {len(bad_paths)} files to rebuild")

    for out_npz in bad_paths:
        try:
            image_key, split, j = parse_bad_path(out_npz)
        except Exception:
            print(f"[SKIP] Unrecognized path pattern: {out_npz}")
            continue

        ep0 = os.path.join(args.data_root, split, f"episode_{j:07d}.npz")
        ep1 = os.path.join(args.data_root, split, f"episode_{j + args.frame_gap:07d}.npz")

        # 检查原始帧是否齐全
        if not os.path.exists(ep0):
            print(f"[WARN] Missing source ep0: {ep0} ; write zeros")
            Nq = grid.shape[1]
            atomic_save_npz(out_npz, tracks=np.zeros((Nq, 2), np.float32), visibility=np.ones((Nq,), bool))
            continue

        if not os.path.exists(ep1):
            # 尾段样本：按原逻辑写零
            print(f"[WARN] Missing source ep1: {ep1} ; write zeros")
            Nq = grid.shape[1]
            atomic_save_npz(out_npz, tracks=np.zeros((Nq, 2), np.float32), visibility=np.ones((Nq,), bool))
            continue

        # 读取两帧
        try:
            img0 = load_image_from_episode(ep0, image_key, (H, W))
            img1 = load_image_from_episode(ep1, image_key, (H, W))
        except Exception as e:
            print(f"[ERROR] Read episode failed: j={j} {e} ; write zeros")
            Nq = grid.shape[1]
            atomic_save_npz(out_npz, tracks=np.zeros((Nq, 2), np.float32), visibility=np.ones((Nq,), bool))
            continue

        # 组装 [B=1, 2, C, H, W]
        video = np.stack([img0, img1], axis=0)              # [2,H,W,C]
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float().unsqueeze(0)  # [1,2,3,H,W]
        video = video.to(device, non_blocking=True)

        with torch.inference_mode():
            pt, pv = model(video, queries=grid, grid_size=grid_size, backward_tracking=False)  # [1,2,Nq,2], [1,2,Nq]

        # Δtracks = last - first；visibility 取末帧
        tracks = (pt[:, 1:2, :, :] - pt[:, 0:1, :, :]).squeeze(1).squeeze(0).detach().cpu().numpy()  # [Nq,2]
        visibility = pv[:, 1, :].squeeze(0).detach().cpu().numpy().astype(bool)                     # [Nq]

        atomic_save_npz(out_npz, tracks=tracks.astype(np.float32), visibility=visibility)
        print(f"[OK] Rebuilt: {out_npz}")

    print("[DONE] All requested files processed.")


if __name__ == "__main__":
    main()
