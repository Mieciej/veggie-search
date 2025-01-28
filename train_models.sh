#!/bin/bash
python3 -m venv .venv
source .vev/bin/activate
pip install -r requirements.txt
curl -L -o ./veggie-images.zip\
  https://www.kaggle.com/api/v1/datasets/download/misrakahmed/vegetable-image-dataset
unzip veggie-images.zip
mv "Vegetable Images" veggie-images
python model_scripts/customnet-train.py
python model_scripts/efficientnet-train.py
python model_scripts/resnet50-train.py
python model_scripts/xception.py

