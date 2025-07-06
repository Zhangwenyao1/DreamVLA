# Running
## Notice

For convenience, some checkpoints, such as the MAE-pretrained ViT-B model, are provided for manual download. Users must update the following paths accordingly. Relevant checkpoints can be acquired from the [website](https://drive.google.com/drive/folders/1F3IE95z2THAQ_lt3DKUFdRGc86Thsnc7?usp=sharing).
* :exclamation: **pretrain.sh, finetune.sh, scratch, eval.sh:**
Please update the following:
    * **calvin_dataset_path** to the directory where you have stored the CALVIN ABC-D data.
    * **save_checkpoint_path** to the parent directory where your experiment checkpoints are saved.  Recommend to create a ```checkpoints``` folder in the project root directory.
    * **finetune_from_pretrained_ckpt** to the location of your pre-trained checkpoint.
    * **resume_from_checkpoint** to the location of your fine-tuned checkpoint.
    * **vit_checkpoint_path** to the location of your ViT checkpoint (downloaded from the [website](https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing)). Recommend to be stored in ```checkpoints/vit_mae/mae_pretrain_vit_base.pth```.

* :exclamation: **networkx:**
Due to compatibility issues between the networkx library in CALVIN and Python 3.10, we provide a compatible version of networkx.zip on the [website](https://drive.google.com/file/d/1z-d1SaI0rXfBtBicw1zPSsP-wE-26oLq/view?usp=sharing). Download and unzip it, then replace the existing networkx library in the following path:

## Seer
### Pre-train
```bash
# Pre-train Seer on Calvin ABC-D dataset
bash scripts/CALVIN_ABC_D/Seer/pretrain.sh
# Pre-train Seer-Large on Calvin ABC-D dataset
bash scripts/CALVIN_ABC_D/Seer-Large/pretrain.sh
```

### Fine-tune
```bash
# Fine-tune Seer on Calvin ABC-D dataset
bash scripts/CALVIN_ABC_D/Seer/finetune.sh
# Fine-tune Seer-Large on Calvin ABC-D dataset
bash scripts/CALVIN_ABC_D/Seer-Large/finetune.sh
```

### Train from Scratch
```bash
# Train Seer on Calvin ABC-D dataset from scratch
bash scripts/CALVIN_ABC_D/Seer/scratch.sh
# Train Seer-Large on Calvin ABC-D dataset from scratch
bash scripts/CALVIN_ABC_D/Seer-Large/scratch.sh
```

### Eval
```bash
# Evaluate Seer on Calvin ABC-D benchmark
bash scripts/CALVIN_ABC_D/Seer/eval.sh
# Evaluate Seer-Large on Calvin ABC-D benchmark
bash scripts/CALVIN_ABC_D/Seer-Large/eval.sh
```

