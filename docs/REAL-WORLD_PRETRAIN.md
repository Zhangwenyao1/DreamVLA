# Pre-train
## Notice
We provide code for pre-training on both the DROID and OXE datasets. Users should update the save_checkpoint_path to the directory where you want to save the training checkpoints, and modify the root_dir to the location where the preprocessed real data is stored. Additionally, users should configure the SLURM information in the provided scripts.

Preparation
```python
cd ${YOUR_PATH_TO_SEER}
conda activate seer
```
## Pre-train (DROID FULL)
* For single-node pre-training:
```bash
bash scripts/REAL/single_node_full_cluster.sh
```
* For multi-node pre-training:
```bash
bash scripts/REAL/slurm_s_full_cluster.sh
```
## Pre-train (DROID with Language)
* For single-node pre-training:
```bash
bash scripts/REAL/single_node_language_cluster.sh
```
* For multi-node pre-training:
```bash
bash scripts/REAL/slurm_s_language_cluster.sh
```
## Pre-train (OXE)
* For multi-node pre-training:
```bash
bash scripts/REAL/slurm_s_language_cluster.sh
```