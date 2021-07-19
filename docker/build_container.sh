#!/bin/bash
NAME=yolo-wt
IMAGE=yolo:latest
MPATH=/home

echo ${NAME}

exec docker run -it --name ${NAME} \
                -v ${MPATH}:${MPATH} \
                ${IMAGE}      \
                bash

