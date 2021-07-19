#!/bin/bash

DATA=coco128.yaml
MODEL_CFG=models/yolov3-tiny2.yaml
WEIGHTS=' '

python train.py --cfg ${MODEL_CFG} \
                --weights ${WEIGHTS} \
                --data ${DATA} \
                --epochs 5 \
                --batch-size 4
                