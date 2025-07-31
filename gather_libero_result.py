import os
import numpy as np
from natsort import natsorted
root = 'eval_libero_large'
result_root = 'eval'

for exp_name in os.listdir(root):
    result_root = os.path.join(root, exp_name)

    all_logs = natsorted(os.listdir(result_root))
    all_results = []
    for log in all_logs:
        with open(os.path.join(result_root, log), 'r') as f:
            lines = f.readlines()
        lines = [item.strip() for item in lines]
        cur_results = []
        for line in lines:
            if line.endswith('%'):
                cur_results.append(float(line[:-1]))
        all_results.append(cur_results)
    # print(all_results)
    mean_sr = [np.mean(item) for item in all_results]
    print(exp_name)
    print(mean_sr)
    print(max(mean_sr))