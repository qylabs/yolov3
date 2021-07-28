#!/bin/bash

# WEIGHTS='weights/yolov3-tiny.pt'
# WEIGHTS='weights/yolov3-tiny2.pt'
WEIGHTS='runs/train/exp/weights/best.pt'
# IMG_SIZE=320
# IMG_SIZE=160,120

IMG_CHANNEL=1 #3


python models/export.py --weights ${WEIGHTS} \
                        --img-size 160 120 \
                        --batch-size 1 \
                        --include onnx \
                        --img-channel ${IMG_CHANNEL} \
                        --simplify  #use onnxsim to simply onnx graph