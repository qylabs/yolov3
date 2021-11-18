#!/bin/bash
:'安装最新版nvidia-docker（现在叫NVIDIA Container Toolkit）
#### 添加软件源
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

#### 安装NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

#### 重启docker
sudo systemctl restart docker

#### 测试NVIDIA Container Toolkit是否安装成功，如果输出显卡驱动信息则安装成功
docker run --gpus all nvidia/cuda:9.0-base nvidia-smi

############
docker pull pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
'

NAME=pytoch_cuda
IMAGE=pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
MPATH=/home

echo ${NAME}

exec docker run -it --name ${NAME} \
                --gpus all \
                -v ${MPATH}:${MPATH} \
                ${IMAGE}      \
                bash

