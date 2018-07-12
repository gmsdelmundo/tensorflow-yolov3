#!/bin/bash

# Download relevant packages
if [[ $1 = "cpu" ]]; then
    sed -i -e "s/tensorflow-gpu/tensorflow/g" requirements.txt
    sed -i -e "s/'NCHW'/'NHWC'/g" config/config.py
fi
pip install -r requirements.txt

# Get the official YOLOv3 weights
cd config
wget https://pjreddie.com/media/files/yolov3.weights
