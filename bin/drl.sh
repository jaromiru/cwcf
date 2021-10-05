#!/usr/bin/env bash

# Parameters
Ds=$1
La=$2
GPU=$3

# Output Path
filepath="/home/gmatlin3/cwcf/output/${Ds}/flambda${La}/"
mkdir -p $filepath

# Job Execution
CUDA_VISIBLE_DEVICES=$3 python -u main.py --dataset $Ds --flambda $La --use_hpc 1 --pretrain 1 \
| tee "${filepath}drl-${Ds}-flambda${La}.log"

# MATLAB Eval Code
cp "/home/gmatlin3/cwcf/tools/debug.m" $filepath