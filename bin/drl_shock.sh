#!/usr/bin/env bash

Ds='shock'
La=1.0
filepath="/home/gmatlin3/cwcf/output/${Ds}/flambda${La}/"
mkdir -p $filepath
CUDA_VISIBLE_DEVICES=7 python -u main.py --dataset $Ds --flambda $La --use_hpc 1 --pretrain 1 \
| tee "${filepath}drl-${Ds}-flambda${La}.log"