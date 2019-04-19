#!/bin/bash

pip install --user -r requirements.txt

python etl.py
start https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/master?filepath=.%2FEDA%2FData%20Exploration.ipynb
python featureEngineering.py
python train.py
