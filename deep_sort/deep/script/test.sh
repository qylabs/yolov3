#!/bin/bash

# DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/market1501
# # CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/ckpt.pt #(128,64) 0.985
# SAVE_FEATURE=./checkpoint/features.pth

# DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/market1501
# CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/resnet18.pt #(128,64) no normalization 0.658
# MODEL_NAME=resnet18
# SAVE_FEATURE=./checkpoint/features_resent18.pth


# DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/market1501
# CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/mobilenetv2_1dot0_market.pth.tar
# MODEL_NAME=mobilenetv2_x1_0
# # SAVE_FEATURE=./checkpoint/features_mobilenetv2_x1_0.pth #(128,64) 0.740
# SAVE_FEATURE=./checkpoint/features_mobilenetv2_x1_0_r256.pth #better but slower (256,128) 0.982

DATA=/home/wangtao/project/yolov3/deep_sort/deep/data/market1501
CHECKPOINT=/home/wangtao/project/yolov3/deep_sort/deep/checkpoint/osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth
MODEL_NAME=osnet_x0_25
SAVE_FEATURE=./checkpoint/features_osnet_x0_25_r256.pth #(256,128) 0.987


python test.py --data-dir ${DATA} \
                --model-name ${MODEL_NAME} \
                --checkpoint ${CHECKPOINT} \
                --save_feature ${SAVE_FEATURE} \
                --num_classes 751