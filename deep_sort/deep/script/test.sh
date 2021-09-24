#!/bin/bash

DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/market1501
CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/ckpt.pt
# SAVE_MODEL=resnet18.pt
# MODEL_NAME=
SAVE_FEATURE=./checkpoint/features.pth

python test.py --data-dir ${DATA} \
                --model-name '' \
                --checkpoint ${CHECKPOINT} \
                --save_feature ${SAVE_FEATURE} \
                --num_classes 751 \