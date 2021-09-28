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

# DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/cust_person_reid_reverse
# CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/resnet18.pt
# MODEL_NAME=resnet18
# SAVE_FEATURE=./checkpoint/features_resnet18_cust_reverse.pth


DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/cust_person_reid_reverse
CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/mobilenetv2_1dot0_market.pth.tar
MODEL_NAME=mobilenetv2_x1_0
# SAVE_FEATURE=./checkpoint/features_mobilenetv2_x1_0_r256_cust_reverse.pth
SAVE_FEATURE=./checkpoint/features_mobilenetv2_x1_0_r128_cust_reverse.pth


# DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/cust_person_reid_reverse
# CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth
# MODEL_NAME=osnet_x0_25
# SAVE_FEATURE=./checkpoint/features_osnet_x0_25_r256_cust_reverse.pth

python test.py --data-dir ${DATA} \
                --model-name ${MODEL_NAME} \
                --checkpoint ${CHECKPOINT} \
                --save_feature ${SAVE_FEATURE} \
                --galler_id_reform 0 \
                --num_classes 751