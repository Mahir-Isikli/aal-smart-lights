FROM python:3.8.13-slim-bullseye

ENV TZ="Europe/Berlin"

RUN DEBIAN_FRONTEND=noninteractive apt update \
    && DEBIAN_FRONTEND=noninteractive apt upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt install tzdata ffmpeg libsm6 libxext6 -y

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip3 install -r requirements.txt \
    && pip3 install opencv-python==4.5.5.64 opencv-contrib-python==4.5.5.64

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CMD python3 activity_detection_prediction.py