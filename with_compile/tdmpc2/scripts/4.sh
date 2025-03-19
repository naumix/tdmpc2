#!/bin/bash 
# 
 
source ~/miniconda3/bin/activate 
conda init bash 
source ~/.bashrc 
conda activate tdmpc2 
 
module load cuDNN/8.9.2.26-CUDA-12.2.0 
python3 train.py task=humanoid9 seed=4