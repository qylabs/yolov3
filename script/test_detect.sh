#!/bin/bash

# WEIGHTS=weights/yolov3-tiny.pt
# SOURCE=data/coco128/images/train2017

# WEIGHTS=runs/train/exp_yolov3_tiny2_WP/weights/best.pt
# SOURCE=data/images
# SOURCE=data/WiderPerson/images/val


# WEIGHTS=runs/train/exp_yolov3_tiny2_gray_WP/weights/best.pt
# SOURCE=data/WiderPerson_gray/images/val
# SOURCE=data/cust_data


WEIGHTS=runs/train/exp_yolov3_tiny3_gray_WP/weights/best.pt
# SOURCE=data/WiderPerson_gray/images/val
# SOURCE=data/cust_data/6_100_results_origin_jpg
# SOURCE=data/cust_data/7_100_results_origin_jpg
# SOURCE=data/cust_data/8_100_results_origin_jpg
SOURCE=data/cust_data/img_3

python3 detect.py --source ${SOURCE} \
                  --weights ${WEIGHTS} \
                  --conf 0.2 \
                  --img-size 320 \
                  --gray_input \
                  --save-crop