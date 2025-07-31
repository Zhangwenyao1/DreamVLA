import os
from natsort import natsorted
import numpy as np
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio
data_root = 'libero_10_converted/episodes'
all_episodes = natsorted(os.listdir(data_root))
all_files = []
os.makedirs("vis/vis_libero_video", exist_ok=True)
for epi_id in tqdm(all_episodes[::10]):
    if epi_id.endswith('.h5'):
        continue
    seq_files = natsorted(os.listdir(os.path.join(data_root, epi_id, 'steps'))) 
    seq_image_primary = []
    seq_image_wrist = []
    for frame_id in seq_files:
        seq_image_primary.append(os.path.join(data_root, epi_id, 'steps', frame_id, 'image_primary.jpg'))
        seq_image_wrist.append(os.path.join(data_root, epi_id, 'steps', frame_id, 'image_wrist.jpg'))
    
    images = []
    for file1, file2 in zip(seq_image_primary, seq_image_wrist):
        image1 = np.asarray(Image.open(file1))[::-1]
        image2 = np.asarray(Image.open(file2))
        images.append(np.concatenate([image1, image2], axis=1))
    
    imageio.mimsave(f'vis/vis_libero_video/{epi_id}.gif', images, 'GIF', fps=30)