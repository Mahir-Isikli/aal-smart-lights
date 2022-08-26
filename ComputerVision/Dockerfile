FROM python:3.8.13-bullseye

# update and upgrade
RUN apt-get update && apt-get upgrade -y
# set timezone
RUN apt update && apt install tzdata -y
ENV TZ="Europe/Berlin"
# install opencv dependencies
RUN apt-get install ffmpeg libsm6 libxext6  -y

# set working directory
WORKDIR /usr/src/app
# copy files into the working directory
COPY . /usr/src/app

# install the requirements
RUN pip3 install -r requirements.txt
RUN pip3 install opencv-python==4.5.5.64

# set env. variables
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# run the code
CMD python3 activity_detection_prediction.py