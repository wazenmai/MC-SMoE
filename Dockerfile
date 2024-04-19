# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
WORKDIR /appn
COPY . .
RUN pip install https://github.com/bigscience-workshop/promptsource/archive/master.tar.gz
RUN pip install -r requirements.txt

# docker build . -t wazenmai/mc-smoe
# docker push wazenmai/mc-smoe:latest
