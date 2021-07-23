#!/bin/bash

# WEIGHTS=weights/yolov3-tiny.pt
# WEIGHTS=runs/train/exp2/weights/best.pt
WEIGHTS=runs/train/exp12/weights/best.pt
SOURCE=data/images
# SOURCE=data/coco128/images/train2017
# SOURCE=data/WiderPerson/images/val

python3 detect.py --source ${SOURCE} \
                  --weights ${WEIGHTS} \
                  --conf 0.2 \
                  --img-size 320