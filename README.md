<!-- <div align="center">
DreamVLA: Vision-Language-Action Models Dream Comprehensive World Knowledge -->
<!-- </div> -->




<h3 align="center" style="font-size:48px; font-weight:bold; color:#9C276A; margin: 0;">
  <a href="https://arxiv.org/abs/2507.04447" style="color:#9C276A; text-decoration: none;">
    DreamVLA: A Vision-Language-Action Model <br> Dreamed with Comprehensive World Knowledge
  </a>
</h3>

<p align="center">
  ⭐ If our project helps you, please give us a star on GitHub to support us!
</p>

<div align="center">

<!-- <p align="center">
  <a href="https://zhangwenyao1.github.io/">Wenyao Zhang</a>*,
  <a href="https://ericliuhhh.github.io/">Hongsi Liu</a>*,
  <a href="https://qizekun.github.io/">Zekun Qi</a>*,
  <a href="https://wangyunnan.github.io/">Yunnan Wang</a>*,
  <a href="#">Xinqiang Yu</a>,
  <a href="https://jzhzhang.github.io/">Jiazhao Zhang</a>,
  <a href="https://runpeidong.web.illinois.edu/">Runpei Dong</a>,
  <a href="https://jiaweihe.com/">Jiawei He</a>,<br>
  <a href="https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en">Zhizheng Zhang</a>,
  <a href="https://hughw19.github.io/">He Wang</a>,
  <a href="https://ericyi.github.io/">Li Yi</a>,
  <a href="https://www.eitech.edu.cn/?p=leader-Wenjun%20Zeng&tid=19&lang=en">Wenjun Zeng</a>,
  <a href="http://home.ustc.edu.cn/~jinxustc/">Xin Jin</a>
</p>
<!-- </div> -->
<p>
  <a href="https://arxiv.org/abs/2507.04447">
    <img src="https://img.shields.io/badge/Paper-PDF-orange.svg" alt="Paper PDF">
  </a>
  <a href="https://zhangwenyao1.github.io/DreamVLA">
    <img src="https://img.shields.io/badge/Project-Page-Green.svg" alt="Project Page">
  </a>
  <a href="https://huggingface.co/WenyaoZhang/DreamVLA">
    <img src="https://img.shields.io/badge/🤗-Hugging_Face-yellow.svg" alt="Hugging Face">
  </a>
  <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg" alt="Code License">
  </a>
  <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE">
    <img src="https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg" alt="Data License">
  </a>
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dreamvla-a-vision-language-action-model/robot-manipulation-on-calvin)](https://paperswithcode.com/sota/robot-manipulation-on-calvin?p=dreamvla-a-vision-language-action-model)
</p>

<p align="center">
If you have any questions about the code, feel free to open an issue!
</p>

</div>


## The difference from previous works
<div style="text-align: center;">
    <img src="assets/paradigm_compare.gif" width=100% >
</div>

## Overall framework of DreamVLA
<div style="text-align: center;">
    <img src="assets/pipeline.gif" width=100% >
</div>


## Clone this repo

```
git clone https://github.com/Zhangwenyao1/DreamVLA
```

This repository's code is based on the [Seer](https://github.com/OpenRobotLab/Seer/tree/main).


## Running on the Benchmark

#### CALVIN ABC-D <a name="calvin abc-d"></a>
- [Installation](docs/CALVIN_ABC-D_INSTALL.md)
- [Running Code](docs/CALVIN_ABC-D_RUN.md)
#### LIBERO <a name="libero"></a>
- [Installation](docs/LIBERO_INSTALL.md)
- [Running Code](docs/LIBERO_RUN.md)


## TODO
- [x] Release the code with LIBERO 



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
