#!/bin/bash

WEIGHTS='weights/yolov3-tiny.pt'
# WEIGHTS='weights/yolov3-tiny2.pt'
IMG_SIZE=320


python models/export.py --weights ${WEIGHTS} \
                        --img ${IMG_SIZE} \
                        --batch-size 1 \
                        --include onnx \
                        --simplify  #use onnxsim to simply onnx graph