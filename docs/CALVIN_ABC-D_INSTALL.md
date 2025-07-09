# Installation

**(1) Conda Env**
```
conda create -n dreamvla python=3.10
conda activate dreamvla
```

**(2) CALVIN**
> Exact instructions on [CALVIN](https://github.com/mees/calvin).
```
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
sh install.sh
```

**(3) Dataset Download**
> We only download CALVIN ABC-D.
```
cd $CALVIN_ROOT/dataset
sh download_data.sh ABC
```

**(4) Third Party Packages**
```
cd ${YOUR_PATH_TO_SEER}
pip install -r requirements.txt
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

**(5) Create a soft link to CALVIN**
```
cd ${YOUR_PATH_TO_SEER}
ln -s $CALVIN_ROOT calvin
```

**(6) Copy the index file `except_lang_idx.npy` to the CALVIN ABC-D training data directory.**
```python
cp -r data_info/auto_lang_ann.npy calvin/dataset/task_ABC_D/training
cp -r data_info/except_lang_idx/except_lang_idx.npy calvin/dataset/task_ABC_D/training
```
