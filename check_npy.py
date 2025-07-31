import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

folder_path = "libero_10_converted/sam_feats"
npy_files = glob.glob(f"{folder_path}/**/*.npy", recursive=True)

def check_npy(npy_file):
    try:
        np.load(npy_file)
        return None  # 没有问题
    except:
        return npy_file  # 返回出错的文件路径

bad_files = []
# for npy_file in npy_files:
#     check_npy(npy_file)

with ThreadPoolExecutor(max_workers=256) as executor:  # 可调整线程数
    futures = [executor.submit(check_npy, f) for f in npy_files]
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if result:
            bad_files.append(result)

print("出错的文件：")
for bf in bad_files:
    print(bf)