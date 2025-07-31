# Installation

**(1) Conda Env**
```
conda create -n dreamvla python=3.10
conda activate dreamvla
```

**(2) LIBERO Env**
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install transformers==4.40.2
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```