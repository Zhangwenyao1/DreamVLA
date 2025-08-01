
# Notice

For convenience, some checkpoints, such as the MAE-pretrained ViT-B model, are provided for manual download. Users must update the following paths accordingly. Relevant checkpoints can be acquired from the [website](https://drive.google.com/drive/folders/1F3IE95z2THAQ_lt3DKUFdRGc86Thsnc7?usp=sharing).
* :exclamation: **pretrain.sh, finetune.sh, scratch, eval.sh:**
Please update the following:
    * **calvin_dataset_path** to the directory where you have stored the CALVIN ABC-D data.
    * **save_checkpoint_path** to the parent directory where your experiment checkpoints are saved.  Recommend to create a ```checkpoints``` folder in the project root directory.
    * **finetune_from_pretrained_ckpt** to the location of your pre-trained checkpoint.
    * **resume_from_checkpoint** to the location of your fine-tuned checkpoint.
    * **vit_checkpoint_path** to the location of your ViT checkpoint (downloaded from the [website](https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing)). Recommend to be stored in ```checkpoints/vit_mae/mae_pretrain_vit_base.pth```.



# Data Processing

Note: there is potential problem that ```Use .reshape(...) instead.```, just change it.

### Dynamic Region:  
Install [co-tracker](https://github.com/facebookresearch/co-tracker.git). Note download the [checkpoints of co-tracker](https://huggingface.co/facebook/cotracker3/blob/main/scaled_offline.pth) and put it to ```./co-tracker/checkpoints```
```.
mv ./data_process/cotrack_extractor.py ./co-tracker/
cd co-tracker
torchrun --nproc_per_node=8 cotrack_extractor.py
```

### SAM Feature: 
Install [SAM](https://github.com/facebookresearch/segment-anything). Note download the [checkpoints of SAM](https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth) and put it to ```./segment-anything/ckpts```.
```
cp dist_utils.py ./segment-anything/
mv ./data_info/ep_start_end_ids.npy <your_data_path>
mv ./data_process/sam_extractor.py ./segment-anything/
cd segment-anything
torchrun --nproc_per_node=8 sam_extractor.py
```

### DINOv2 Feature: 

Install [DINOV2](https://github.com/facebookresearch/dinov2). Note download the [checkpoints of dinov2]( https://huggingface.co/junjiexv/dinov2_vit/blob/main/dinov2_vits14_pretrain.pth) and put it to ```./dinov2/ckpts```.
```
cp dist_utils.py ./dinov2/
mv ./data_process/dino_extractor.py ./dinov2/
cd dinov2
torchrun --nproc_per_node=8 dino_extractor.py
```
If you want to finetune our model, ```python dino_extractor.py``` is must to run.

### Optional

To reduce I/O overhead, you can merge all processed data together with the raw CALVIN dataset into a single package that contains RGB, depth, and semantic‐label files.
If you choose this option, remember to add the flag --merge_data in both finetune.sh and scratch.sh (see scripts/CALVIN_ABC_D/).finetune_merge_data.sh```

```
python ./data_process/merge_sam_dino.py # merge sam and dino feature into new dataset
python ./data_process/merge_track.py # merge optical flow into new dataset
```






# Run
Note: you need to change the detail of the *.sh in ```./scripts/CALVIN_ABC_D/DreamVLA/```. Moreover, if you use less than 8 gpus, plase change the *node_num* in *.sh.


### Pretrain:
```
bash ./scripts/CALVIN_ABC_D/DreamVLA/pretrain.sh
```
You can also load the pretrained weights from [pretrained seer-large](https://drive.google.com/drive/folders/1AFabqfDEi69oMo0FTGhEiH2QSRLYBR9r).
In our experiments they deliver the same performance, so you can use either set interchangeably.

### Finetune:
```
bash ./scripts/CALVIN_ABC_D/DreamVLA/finetune.sh
```



### Evaluation
Download our [checkpoint](https://drive.google.com/drive/folders/1P1fA2vTUF-lsrrWyNvDSWE1ATTHbQQ9T?usp=drive_link) and create ```checkpoints/```. Then put it into the file.
```
bash ./scripts/CALVIN_ABC_D/DreamVLA/eval.sh
```

