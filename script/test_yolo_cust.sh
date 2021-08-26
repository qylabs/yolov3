#!/bin/bash

DATA=data/cust_data.yaml
WEIGHTS=runs/train/exp_yolov3_tiny3_gray_WP/weights/best.pt
HYP=data/hyp.scratch_gray.yaml
IMG_SIZE=160


python test.py  --weights ${WEIGHTS} \
                --data ${DATA} \
                --batch-size 16 \
                --img-size ${IMG_SIZE} \
                --conf-thres 0.2 \
                --hyp ${HYP}
                # > log.file 2>&1 &