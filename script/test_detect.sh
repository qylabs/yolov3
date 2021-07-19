#!/bin/bash

WEIGHTS=weights/yolov3-tiny.pt

python3 detect.py --source data/images \
                  --weights ${WEIGHTS} \
                  --conf 0.25