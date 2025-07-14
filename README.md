<!-- <div align="center">
DreamVLA: Vision-Language-Action Models Dream Comprehensive World Knowledge -->
<!-- </div> -->

<h5 align="center" style="font-size:20px;"><a href="https://arxiv.org/abs/2507.04447" style="color:#9C276A" >
DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge</a></h5>

<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<!-- # DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge -->

[Wenyao Zhang](https://zhangwenyao1.github.io/)\*, [Hongsi Liu](https://ericliuhhh.github.io/)\*, [Zekun Qi](https://qizekun.github.io/)\*, [Yunnan Wang](https://wangyunnan.github.io/)\*, [Xinqiang Yu](), [Jiazhao Zhang](https://jzhzhang.github.io/), [Runpei Dong](https://runpeidong.web.illinois.edu/), [Jiawei He](https://jiaweihe.com/), [Zhizheng Zhang](https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en), [He Wang](https://hughw19.github.io/), [Li Yi](https://ericyi.github.io/), [Wenjun Zeng](https://www.eitech.edu.cn/?p=leader-Wenjun%20Zeng&tid=19&lang=en), [Xin Jin](http://home.ustc.edu.cn/~jinxustc/).

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-orange.svg)](https://arxiv.org/abs/2507.04447)
[![Project Page](https://img.shields.io/badge/Project-Page-Green.svg)](https://zhangwenyao1.github.io/DreamVLA)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging_Face-yellow.svg)](https://huggingface.co/WenyaoZhang/DreamVLA)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)


## The difference from previous works
<div style="text-align: center;">
    <img src="assets/paradigm_compare.gif" width=100% >
</div>

## Overall framework of DreamVLA
<div style="text-align: center;">
    <img src="assets/pipeline.gif" width=100% >
</div>

# 


# Table of Contents:
1. [Installaton](#installation)
2. [Data Processing](#data)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [License](#license)
6. [Citation](#citation).
7. [Acknowledgment](#acknowledgment)


# Installation

### Create an anaconda environment

```
conda create -n dreamvla python=3.10
conda activate dreamvla
```

### Clone this repo

```
git clone https://github.com/Zhangwenyao1/DreamVLA
```

This repository's code is based on the [Seer](https://github.com/OpenRobotLab/Seer/tree/main).


### Install for CALVIN

- [Installation](docs/CALVIN_ABC-D_INSTALL.md)
- [Running Code](docs/CALVIN_ABC-D_RUN.md)


# Data Processing

### Dynamic Region:  
Install [co-tracker](https://github.com/facebookresearch/co-tracker.git)
```
mv ./data_process/cotrack_extractor.py ./co-tracker/
cd co-tracker
python cotrack_extractor.py
```

### SAM Feature: 
Install [SAM](https://github.com/facebookresearch/segment-anything)
```
mv ./data_process/sam_extractor.py ./segment-anything/
cd segment-anything
python sam_extractor.py
```

### DINOv2 Feature: 

Install [DINOV2](https://github.com/facebookresearch/dinov2)
```
mv ./data_process/dino_extractor.py ./dinov2/
cd dinov2
python dino_extractor.py
```

Merge all data and raw calvin dataset to produce the new dataset
```
python ./data_process/merge_sam_dino.py # merge sam and dino feature into new dataset
python ./data_process/merge_track.py # merge optical flow into new dataset
```


# Training
### Pretrain:
```
bash ./scripts/CALVIN_ABC_D/DreamVLA/pretrain.sh
```

### Finetune:
```
bash ./scripts/CALVIN_ABC_D/DreamVLA/finetune.sh
```


# Evaluation

Down load our [checkpoint](https://drive.google.com/drive/folders/1P1fA2vTUF-lsrrWyNvDSWE1ATTHbQQ9T?usp=drive_link)
```
bash ./scripts/CALVIN_ABC_D/DreamVLA/eval.sh
```



## Acknowledgement

We would like to express our deepest gratitude to [Yang Tian](https://scholar.google.com/citations?user=leXXHKwAAAAJ&hl=zh-CN) for the technique support!!!

## Citation

If you find our ideas / environments helpful, please cite our work at

```
article{dreamvla25,
          author = {Wenyao Zhang and
                    Hongsi Liu and
                    Zekun Qi and
                    Yunan Wang and
                    Xinqiang Yu and
                    Jiazhao Zhang and
                    Runpei Dong and
                    Jiawei He and
                    He Wang and
                    Zhizheng Zhang and
                    Li Yi and 
                    Wenjun Zeng and
                    Xin Jin},
          title        = {DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge},
          journal      = {CoRR},
          volume       = {abs/2507.04447},
          year         = {2025},
          url          = {https://doi.org/10.48550/arXiv.2507.04447},
          doi          = {10.48550/ARXIV.2507.04447},
          eprinttype    = {arXiv},
          eprint       = {2507.04447}
        }
```
