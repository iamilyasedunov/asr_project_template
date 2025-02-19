# Source of image: https://hub.docker.com/r/nvidia/cuda
# Discussion about CUDA vesrion https://discuss.pytorch.org/t/pytorch-with-cuda-11-compatibility/89254
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Install prerequested
RUN apt-get -y update && \
    apt-get -y install vim \
                       htop \
                       git \
                       wget \
                       sudo \
                       software-properties-common \
                       unzip \
                       tmux \
                       tree \
                       bash-completion \
                       nano

# System requirements
RUN apt-get -y install libsndfile1

# Install anaconda from https://repo.anaconda.com/archive/
RUN . ~/.bashrc && \
    wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda3.sh && \
    bash ~/anaconda3.sh -b && \
    echo 'export PATH="/root/anaconda3/bin:$PATH"' >> ~/.bashrc && \
    rm ~/anaconda3.sh

# Set normal TZ (mostly for logs)
RUN ln -sf /usr/share/zoneinfo/Europe/Moscow /etc/localtime

# Pytorch
# By default install latest version:
# Version downgrading available via requirements.txt
RUN . ~/.bashrc && conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Requirements
COPY requirements.txt /root/requirements.txt
# RUN pip install --upgrade pip
RUN . ~/.bashrc && pip install --upgrade pip && pip install -r /root/requirements.txt && \
                                            pip install --ignore-installed ruamel.yaml && \
                                            pip install https://github.com/kpu/kenlm/archive/master.zip && \
                                            pip install pyctcdecode==0.1.1
# RUN pip install --ignore-installed ruamel.yaml