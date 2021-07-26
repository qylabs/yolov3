#!/bin/bash

DATA=data/WiderPerson_gray.yaml
MODEL_CFG=models/yolov3-tiny2_gray.yaml
WEIGHTS=' '
EPOCHS=300
IMG_SIZE=320
HYP=data/hyp.scratch_gray.yaml

python train.py --cfg ${MODEL_CFG} \
                --weights '' \
                --data ${DATA} \
                --epochs ${EPOCHS} \
                --batch-size 16 \
                --img-size ${IMG_SIZE} \
                --workers 0 \
                --hyp ${HYP}
                # > log.file 2>&1 &
                