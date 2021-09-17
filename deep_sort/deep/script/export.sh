#!/bin/bash

WEIGHTS='checkpoint/ckpt.pt'
# IMG_SIZE=128,160 #height,width

IMG_CHANNEL=3 #3


python export.py --weights ${WEIGHTS} \
                --img-size 128 64 \
                --batch-size 1 \
                --in_channel ${IMG_CHANNEL} \
                --num_classes 751 \
                --simplify  #use onnxsim to simply onnx graph