#!/bin/bash

# DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/market1501
# # CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/ckpt.pt
# SAVE_FEATURE=./checkpoint/features.pth

DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/market1501
CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/resnet18.pt
MODEL_NAME=resnet18
SAVE_FEATURE=./checkpoint/features_resent18.pth


python test.py --data-dir ${DATA} \
                --model-name ${MODEL_NAME} \
                --checkpoint ${CHECKPOINT} \
                --save_feature ${SAVE_FEATURE} \
                --num_classes 751 \