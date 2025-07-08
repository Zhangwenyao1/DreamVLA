# Installation
To set up Seer, we will create an isolated environment called seer. This environment is designed to support pre-training, fine-tuning, and inference workflows.

## seer env
**(1) Env**
```python
conda create -n seer python=3.10
conda activate seer
```
**(2) Third Party Packages**
```python
cd ${YOUR_PATH_TO_SEER}
pip install -r requirements.txt
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```
