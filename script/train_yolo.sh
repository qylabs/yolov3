#!/bin/bash

DATA=data/coco128.yaml
MODEL_CFG=models/yolov3-tiny2.yaml
WEIGHTS=' '
EPOCHS=300

python train.py --cfg ${MODEL_CFG} \
                --weights '' \
                --data ${DATA} \
                --epochs ${EPOCHS} \
                --batch-size 16 \
                > log.file 2>&1 &
                