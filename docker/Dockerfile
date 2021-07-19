FROM ubuntu:20.04
# FROM daocloud.io/library/ubuntu:20.04 #local ubuntu:20.04 images

#MAINTAINER wangtao

ENV DEBIAN_FRONTEND=noninteractive

#/*OS Requirements installation*/
RUN apt-get update && \
    apt-get install -y build-essential \
                        git \
                        libftdi-dev \
                        libftdi1 doxygen \
                        python3-pip \
                        libsdl2-dev \
                        curl \
                        wget \
                        cmake \
                        libusb-1.0-0-dev \
                        scons \
                        gtkwave \
                        libsndfile1-dev \
                        rsync \
                        autoconf \
                        automake \
                        texinfo \
                        libtool \
                        pkg-config \
                        libsdl2-ttf-dev \
                        libjpeg-dev \
                        vim \
                        udev \
        && apt-get clean \
        && apt-get autoclean
        

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10


# #copy other python requirements
COPY requirements.txt .

RUN pip3 install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple



