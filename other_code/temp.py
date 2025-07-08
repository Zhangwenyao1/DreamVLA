import os

# def get_largest_file(path):
#     max_size = -1
#     max_file = None

#     for root, _, files in os.walk(path):
#         for name in files:
#             try:
#                 file_path = os.path.join(root, name)
#                 size = os.path.getsize(file_path)
#                 if size > max_size:
#                     max_size = size
#                     max_file = file_path
#             except Exception as e:
#                 print(f"Error accessing {file_path}: {e}")

#     return max_file, max_size
# file_path, size = get_largest_file("/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/rgb_gripper/training")
# print(f"最大文件: {file_path}")
# print(f"大小: {size / (1024 * 1024):.2f} MB")


# 示例：写入 LMDB
# import lmdb, torch, os, pickle

# import os
# import lmdb
# import torch
# import pickle
# from tqdm import tqdm

# def pack_pt_to_lmdb(pt_dir, lmdb_path, map_size=1 << 40):  # 默认 1 TB 空间
#     env = lmdb.open(lmdb_path, map_size=map_size)

#     files = sorted([
#         f for f in os.listdir(pt_dir)
#         if f.endswith('.pt')
#     ])

#     with env.begin(write=True) as txn:
#         for fname in tqdm(files, desc="Packing .pt to LMDB"):
#             key = fname.split('.')[0].encode()
#             path = os.path.join(pt_dir, fname)
#             try:
#                 obj = torch.load(path)
#                 data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
#                 txn.put(key, data)
#             except Exception as e:
#                 print(f"Failed to process {fname}: {e}")
    
#     env.close()
#     print(f"✅ 打包完成，LMDB 保存于: {lmdb_path}")

# pack_pt_to_lmdb('/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/rgb_static/training', 'calvin_dino_static_training.lmdb')
# pack_pt_to_lmdb('/mnt/afs/wyzhang/code/segment-anything/calvin_sam/task_ABC_D/rgb_gripper/training', 'calvin_sam_gripper_training.lmdb')


# import os
# import torch
# import numpy as np
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor


# def convert_one_file(fname, gripper_dir, static_dir, output_dir):
#     base_name = fname.split('.')[0]
#     npz_path = os.path.join(output_dir, f"{base_name}.npz")

#     if os.path.exists(npz_path):
#         return f"⏩ Skipped {base_name}"

#     gripper_path = os.path.join(gripper_dir, fname)
#     static_path = os.path.join(static_dir, fname)

#     try:
#         gripper_tensor = torch.load(gripper_path)
#         static_tensor = torch.load(static_path)

#         np.savez_compressed(
#             npz_path,
#             gripper=gripper_tensor.to(torch.float32).numpy(),
#             static=static_tensor.to(torch.float32).numpy()
#         )
#         return f"✅ Saved {base_name}"
#     except Exception as e:
#         return f"❌ Failed {base_name}: {e}"


# def pack_pt_to_npz_parallel(gripper_dir, static_dir, output_dir, num_threads=8):
#     os.makedirs(output_dir, exist_ok=True)

#     gripper_files = sorted([f for f in os.listdir(gripper_dir) if f.endswith('.pt')])
#     static_files = sorted([f for f in os.listdir(static_dir) if f.endswith('.pt')])
#     common_files = sorted(list(set(gripper_files) & set(static_files)))

#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         futures = [executor.submit(convert_one_file, fname, gripper_dir, static_dir, output_dir)
#                    for fname in common_files]

#         for f in tqdm(futures, desc=f"Packing .pt to .npz (threads={num_threads})"):
#             tqdm.write(f.result())

#     print(f"✅ 所有 .npz 保存完成于: {output_dir}")



# 示例用法：
# pack_pt_to_npz('/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/rgb_static/training', '/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/rgb_gripper/training', '/path/to/output_npz')
# pack_pt_to_npz_parallel(
#     gripper_dir="/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/rgb_gripper/training",
#     static_dir="/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/rgb_static/training",
#     output_dir="/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/training",
#     num_threads=16  # 你可以根据机器核数调整
# )

# import os
# import torch
# import numpy as np

# # 指定文件路径
# gripper_path = "/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/rgb_gripper/training/467709.pt"
# static_path = "/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/rgb_static/training/467709.pt"
# output_path = "/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/training/467709.npz"

# try:
#     # 加载并转 float32
#     gripper_tensor = torch.load(gripper_path).to(torch.float32)
#     static_tensor = torch.load(static_path).to(torch.float32)

#     # 保存为 .npz
#     np.savez_compressed(output_path, gripper=gripper_tensor.numpy(), static=static_tensor.numpy())

#     print(f"✅ 成功保存为: {output_path}")

# except Exception as e:
#     print(f"❌ 出错: {e}")



import os
import torch
import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def process_file(args):
    gripper_path, static_path = args
    base_name = os.path.splitext(os.path.basename(gripper_path))[0]
    try:
        gripper_tensor = torch.load(gripper_path).to(torch.float32).numpy()
        static_tensor = torch.load(static_path).to(torch.float32).numpy()
        return base_name, gripper_tensor, static_tensor
    except Exception as e:
        print(f"❌ Failed to process {base_name}: {e}")
        return None

def convert_pt_to_h5_parallel(gripper_dir, static_dir, output_path, num_workers=32):
    gripper_files = sorted([f for f in os.listdir(gripper_dir) if f.endswith('.pt')])
    static_files = sorted([f for f in os.listdir(static_dir) if f.endswith('.pt')])
    common_files = sorted(set(gripper_files) & set(static_files))

    task_list = [
        (os.path.join(gripper_dir, fname), os.path.join(static_dir, fname))
        for fname in common_files
    ]

    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(process_file, task_list), total=len(task_list)):
            if result is not None:
                results.append(result)

    # 将结果写入 HDF5 文件（串行执行，保证线程安全）
    with h5py.File(output_path, 'w') as h5f:
        for base_name, gripper_array, static_array in results:
            grp = h5f.create_group(base_name)
            grp.create_dataset("gripper", data=gripper_array, compression="gzip")
            grp.create_dataset("static", data=static_array, compression="gzip")

    print(f"✅ HDF5 file saved at: {output_path}")


convert_pt_to_h5_parallel(
    gripper_dir="/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/rgb_gripper/training",
    static_dir="/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/rgb_static/training",
    output_path="/mnt/afs/wyzhang/code/dinov2/calvin_dino/task_ABC_D/training_data.h5"
)
