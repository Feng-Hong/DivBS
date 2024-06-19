<h1 align="center">Diversified Batch Selection for Training Acceleration</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2406.04872"><img src="https://img.shields.io/badge/arXiv-2406.04872-b31b1b.svg" alt="Paper"></a>
    <a href="https://openreview.net/pdf?id=5QWKec0eDF"><img src="https://img.shields.io/badge/OpenReview-ICML'24-blue" alt="Paper"></a>
    <a href="https://github.com/Feng-Hong/DivBS"><img src="https://img.shields.io/badge/Github-DivBS-brightgreen?logo=github" alt="Github"></a>
    <!-- <a href="https://iclr.cc/media/iclr-2023/Slides/11305.pdf"> <img src="https://img.shields.io/badge/Slides (5 min)-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a> -->
    <!-- <a href="https://iclr.cc/media/PosterPDFs/ICLR%202023/11305.png?t=1680238646.843235"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a> -->
</p>

by Feng Hong, Yueming Lyu, Jiangchao Yao, Ya Zhang, Ivor W. Tsang, and Yanfeng Wang at SJTU, A*STAR, Shanghai AI Lab, and NTU.

International Conference on Machine Learning (ICML), 2024.

This repository is the official Pytorch implementation of DivBS.

<!-- ⚠️ This repository is currently in its initial version. It is being organized and updated continuously. Please note that this version is not the final release. -->
⚠️ This repository is being organized and updated continuously. Please note that this version is not the final release.

## Citation

If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@inproceedings{
hong2024diversified,
title={Diversified Batch Selection for Training Acceleration},
author={Feng Hong and Yueming Lyu and Jiangchao Yao and Ya Zhang and Ivor Tsang and Yanfeng Wang},
booktitle={ICML},
year={2024}
}
```

## Environment
Create the environment for running our code:
```bash
conda create --name DivBS python=3.7.10
conda activate DivBS
pip install -r requirements.txt
```

## Data Preparation
For CIFAR datasets, the data will be automatically downloaded by the code. 

For Tiny-ImageNet, please download the dataset from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and unzip it to the `_TINYIMAGENET` folder. Then, run the following command to prepare the data:
```bash
cd _TINYIMAGENET
python val_folder.py
```

## Running
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/cifar10_DivBS_01.yaml --seed 0 --wandb_not_upload 
```
The `--wandb_not_upload` is optional and is used to keep wandb log files locally without uploading them to the wandb cloud.

## Contact
If you have any problem with this code, please feel free to contact **feng.hong@sjtu.edu.cn**.
