FROM bitnami/pytorch:2.0.0-debian-11-r3
WORKDIR /app

COPY ./src ./src
COPY ./main.py ./main.py
COPY ./predict.py ./predict.py
COPY ./predict.py ./predict.py
COPY ./requirements.txt ./requirements.txt

RUN mkdir public

# pretrained model
COPY ./pretrained/bottle-cap-20230410044128.cpu.pth ./pretrained/bottle-cap-20230410044128.cpu.pth

ENV DEBIAN_FRONTEND noninteractive

USER root
RUN apt-get update -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 3000

CMD uvicorn main:app --host 0.0.0.0 --port 3000