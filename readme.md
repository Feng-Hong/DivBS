<!-- # Long-Tailed Partial Label Learning via Dynamic Rebalancing -->
<h1 align="center">Diversified Batch Selection for Training Acceleration</h1>

by Feng Hong, Feng Hong, Yueming Lyu, Jiangchao Yao, Ya Zhang, Ivor W. Tsang, and Yanfeng Wang at SJTU, A*STAR, Shanghai AI Lab, and NTU.

International Conference on Machine Learning (ICML), 2024.

This repository is the official Pytorch implementation of DivBS.

⚠️ This repository is currently in its initial version. It is being organized and updated continuously. Please note that this version is not the final release.

## Running the code
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/cifar10_DivBS_01.yaml --seed 0 --wandb_not_upload 
```
The `--wandb_not_upload` is optional and is used to keep wandb log files locally without uploading them to the wandb cloud.

## Contact
If you have any problem with this code, please feel free to contact **feng.hong@sjtu.edu.cn**.