# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
WORKDIR /app
RUN pip install https://github.com/bigscience-workshop/promptsource/archive/master.tar.gz
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .