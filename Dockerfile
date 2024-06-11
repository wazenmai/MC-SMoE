# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app
RUN apt update && apt install -y git wget
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY promptsource promptsource
RUN pip install -e promptsource/
COPY lm-evaluation-harness lm-evaluation-harness
RUN pip install -e lm-evaluation-harness/
COPY . .
