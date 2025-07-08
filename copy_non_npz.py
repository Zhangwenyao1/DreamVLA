import os
import shutil
from tqdm import tqdm

# A和B文件夹路径
src_folder = 'calvin_datasets/task_ABC_D/training'
dst_folder = 'calvin_datasets_with_dinosam/task_ABC_D/training'


os.makedirs(dst_folder, exist_ok=True)

with os.scandir(src_folder) as entries:
    for entry in tqdm(entries):
        # 跳过 .npz 文件
        if entry.is_file() and entry.name.endswith('.npz'):
            continue

        src_path = entry.path
        dst_path = os.path.join(dst_folder, entry.name)

        if entry.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)