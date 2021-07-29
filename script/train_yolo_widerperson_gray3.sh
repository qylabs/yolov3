#!/bin/bash

DATA=data/WiderPerson_gray.yaml
MODEL_CFG=models/yolov3-tiny3_gray.yaml
WEIGHTS=runs/train/exp/weights/best.pt
EPOCHS=200
IMG_SIZE=320
HYP=data/hyp.scratch_gray.yaml

python train.py --cfg ${MODEL_CFG} \
                --weights ${WEIGHTS} \
                --data ${DATA} \
                --epochs ${EPOCHS} \
                --batch-size 16 \
                --img-size ${IMG_SIZE} \
                --workers 0 \
                --hyp ${HYP}
                # > log.file 2>&1 &
                