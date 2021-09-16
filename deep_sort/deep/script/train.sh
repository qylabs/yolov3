#!/bin/bash

DATA=/home/wangtao/project/data/market1501
# CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/ckpt.t7
SAVE_MODEL=resnet18.pth

python train.py --data-dir ${DATA} \
                --model-name 'resnet18' \
                --save-model ${SAVE_MODEL} \
                --epochs 1 
                # --resume \
                # --checkpoint ${CHECKPOINT}