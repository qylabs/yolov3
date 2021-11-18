#!/bin/bash

# WEIGHTS='checkpoint/ckpt.pt'
WEIGHTS='checkpoint/mobilenetv2_1dot0_market.pth.tar'
# IMG_SIZE=128,160 #height,width
MODEL_NAME=mobilenetv2_x1_0

IMG_CHANNEL=3 #3


python export.py --weights ${WEIGHTS} \
                --img-size 128 64 \
                --batch-size 1 \
                --in_channel ${IMG_CHANNEL} \
                --num_classes 751 \
                --model-name ${MODEL_NAME} \
                --simplify  #use onnxsim to simply onnx graph