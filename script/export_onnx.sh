#!/bin/bash

# WEIGHTS='weights/yolov3-tiny.pt'
# WEIGHTS='weights/yolov3-tiny2.pt'
WEIGHTS='runs/train/exp_yolov3_tiny3_gray_WP/weights/best.pt'
# IMG_SIZE=320
# IMG_SIZE=128,160 #height,width

IMG_CHANNEL=1 #3


python models/export.py --weights ${WEIGHTS} \
                        --img-size 128 160 \
                        --batch-size 1 \
                        --include onnx \
                        --img-channel ${IMG_CHANNEL} \
                        --simplify  #use onnxsim to simply onnx graph