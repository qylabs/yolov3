#!/bin/bash

DATA=data/WiderPerson.yaml
MODEL_CFG=models/yolov3-tiny2.yaml
WEIGHTS=' '
EPOCHS=300
IMG_SIZE=320

python train.py --cfg ${MODEL_CFG} \
                --weights '' \
                --data ${DATA} \
                --epochs ${EPOCHS} \
                --batch-size 16 \
                --img-size ${IMG_SIZE}
                # > log.file 2>&1 &
                