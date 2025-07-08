# Quick Training
Preparation
```python
cd ${YOUR_PATH_TO_SEER}
conda activate seer
```
Download related checkpoints from the [checkpoint repository](https://drive.google.com/drive/folders/1rT8JKLhJGIo97jfYUm2JiFUrogOq-dgJ?usp=drive_link).
## :sparkles: Fine-tuning
* For single-node fine-tuning:
```bash
bash scripts/REAL/single_node_ft.sh
```
## :sparkles: Training from Scratch
* For single-node training from scratch:
```bash
bash scripts/REAL/single_node_scratch.sh
```
    