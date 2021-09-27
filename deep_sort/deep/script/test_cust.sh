#!/bin/bash

# DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/cust_person_reid
# CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/ckpt.pt
# # SAVE_MODEL=resnet18.pt
# # MODEL_NAME=
# SAVE_FEATURE=./checkpoint/features_cust.pth


# DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/cust_person_reid_reverse
# CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/ckpt.pt
# SAVE_FEATURE=./checkpoint/features_cust_reverse.pth

# DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/cust_person_reid
# CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/resnet18.pt
# MODEL_NAME=resnet18
# SAVE_FEATURE=./checkpoint/features_resnet18_cust.pth

DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/cust_person_reid_reverse
CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/resnet18.pt
MODEL_NAME=resnet18
SAVE_FEATURE=./checkpoint/features_resnet18_cust_reverse.pth


python test.py --data-dir ${DATA} \
                --model-name ${MODEL_NAME} \
                --checkpoint ${CHECKPOINT} \
                --save_feature ${SAVE_FEATURE} \
                --num_classes 751 \
                --galler_id_reform 0