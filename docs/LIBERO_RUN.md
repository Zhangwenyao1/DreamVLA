# Running
## Notice

For convenience, some checkpoints, such as the MAE-pretrained ViT-B model, are provided for manual download. Users must update the following paths accordingly. Relevant checkpoints can be acquired from the [website](https://drive.google.com/drive/folders/1zwqGvKKtjyuWdDaNSLVGJprJMPoSqAPk?usp=drive_link).
* :exclamation: **pretrain.sh, finetune.sh, scratch, eval.sh:**
Please update the following:
    * **save_checkpoint_path** to the parent directory where your experiment checkpoints are saved.  Recommend to create a ```checkpoints``` folder in the project root directory.
    * **finetune_from_pretrained_ckpt** to the location of your pre-trained checkpoint.
    * **resume_from_checkpoint** to the location of your fine-tuned checkpoint.
    * **vit_checkpoint_path** to the location of your ViT checkpoint (downloaded from the [website](https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing)). Recommend to be stored in ```checkpoints/vit_mae/mae_pretrain_vit_base.pth```.
    * **libero_path** to the location of LIBERO dir.

# Data Processing
### Convert Data
modify the *src_dir* (e.g. libero_10, libero_spatial...) and *tgt_dir* in `utils/convert_libero_per_step.py`, then run
```bash
python utils/convert_libero_per_step.py
```
It will generate converted data in *tgt_dir*.

### Dynamic Region:  
Install [co-tracker](https://github.com/facebookresearch/co-tracker.git). Note download the [checkpoints of co-tracker](https://huggingface.co/facebook/cotracker3/blob/main/scaled_offline.pth) and put it to ```./co-tracker/checkpoints```
```bash
mv ./data_process/cotrack_extractor.py ./co-tracker/
cd co-tracker
torchrun --nproc_per_node=8 cotrack_extractor_libero.py --data_root ${tgt_dir}/episodes --save_path ${tgt_dir}/cotracker_traj
```

### SAM Feature: 
Install [SAM](https://github.com/facebookresearch/segment-anything). Note download the [checkpoints of SAM](https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth) and put it to ```./segment-anything/ckpts```.
```bash
cp dist_utils.py ./segment-anything/
mv ./data_info/ep_start_end_ids.npy <your_data_path>
mv ./data_process/sam_extractor.py ./segment-anything/
cd segment-anything
torchrun --nproc_per_node=8 sam_extractor_libero.py --data_root ${tgt_dir}/episodes --save_path ${tgt_dir}/sam_feats
```

### DINOv2 Feature: 

Install [DINOV2](https://github.com/facebookresearch/dinov2). Note download the [checkpoints of dinov2]( https://huggingface.co/junjiexv/dinov2_vit/blob/main/dinov2_vits14_pretrain.pth) and put it to ```./dinov2/ckpts```.
```bash
cp dist_utils.py ./dinov2/
mv ./data_process/dino_extractor.py ./dinov2/
cd dinov2
torchrun --nproc_per_node=8 dino_extractor_libero.py --data_root ${tgt_dir}/episodes --save_path ${tgt_dir}/dinov2_feats
```

# Training
### Pre-train
```bash
# Pre-train DreamVLA on LIBERO-90 dataset
bash scripts/LIBERO/DreamVLA/pretrain.sh
```
You also can load the pretrained weights from 

### Fine-tune
```bash
# Fine-tune DreamVLA on LIBERO dataset
bash scripts/LIBERO/DreamVLA/finetune_long.sh
bash scripts/LIBERO/DreamVLA/finetune_object.sh
bash scripts/LIBERO/DreamVLA/finetune_spatial.sh
bash scripts/LIBERO/DreamVLA/finetune_goal.sh
```

### Train from Scratch
```bash
# Train DreamVLA on LIBERO dataset from scratch
bash scripts/LIBERO/DreamVLA/scratch_long.sh
bash scripts/LIBERO/DreamVLA/scratch_object.sh
bash scripts/LIBERO/DreamVLA/scratch_spatial.sh
bash scripts/LIBERO/DreamVLA/scratch_goal.sh
```

### Eval
```bash
# Evaluate DreamVLA on LIBERO benchmark
bash scripts/LIBERO/DreamVLA/eval_long.sh
bash scripts/LIBERO/DreamVLA/eval_object.sh
bash scripts/LIBERO/DreamVLA/eval_spatial.sh
bash scripts/LIBERO/DreamVLA/eval_goal.sh
```