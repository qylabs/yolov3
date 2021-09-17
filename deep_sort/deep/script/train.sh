#!/bin/bash

DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/market1501
# CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/ckpt.t7
SAVE_MODEL=resnet18.pt

python train.py --data-dir ${DATA} \
                --model-name 'resnet18' \
                --save-model ${SAVE_MODEL} \
                --epochs 40
                # --resume \
                # --checkpoint ${CHECKPOINT}