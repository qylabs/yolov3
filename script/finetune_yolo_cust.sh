#!/bin/bash

DATA=data/cust_data.yaml
MODEL_CFG=models/yolov3-tiny3_gray.yaml
WEIGHTS=runs/train/exp_yolov3_tiny3_gray_WP/weights/best.pt
EPOCHS=100
IMG_SIZE=160
HYP=data/hyp.finetune_gray.yaml #change from hyp.scratch.yaml to hyp.finetune.yaml

python train.py --cfg ${MODEL_CFG} \
                --weights ${WEIGHTS} \
                --data ${DATA} \
                --epochs ${EPOCHS} \
                --batch-size 16 \
                --img-size ${IMG_SIZE} \
                --workers 0 \
                --hyp ${HYP}
                # > log.file 2>&1 &
                