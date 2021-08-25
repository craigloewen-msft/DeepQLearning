#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------
# FROM tensorflow/tensorflow:latest-gpu
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install python-opengl git openssh-client openssh-server -y 

RUN pip install -r /app/requirements.txt

RUN mkdir tf_train_breakout

CMD ./doTraining.sh

