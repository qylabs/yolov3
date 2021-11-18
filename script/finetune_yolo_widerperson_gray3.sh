#!/bin/bash

DATA=data/qyhit_data.yaml
MODEL_CFG=models/yolov3-tiny3_gray.yaml
WEIGHTS=runs/train/baseline_relu/weights/last.pt
EPOCHS=100
IMG_SIZE=320
HYP=data/hyp.finetune_gray.yaml #change from hyp.scratch.yaml to hyp.finetune.yaml

python train.py --cfg ${MODEL_CFG} \
                --weights ${WEIGHTS} \
                --data ${DATA} \
                --epochs ${EPOCHS} \
                --batch-size 16 \
                --img-size ${IMG_SIZE} \
                --hyp ${HYP}  \
                --single-cls
                # > log.file 2>&1 &
                