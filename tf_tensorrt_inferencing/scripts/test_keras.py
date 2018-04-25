#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from imutils import paths
import argparse
import imutils
import cv2
import os

import sys
sys.path.append("scripts")
from conf import config

def test_keras_model():
    print("Loading network...")
    model = load_model(config['model_file'])

    datagen = ImageDataGenerator()

    generator = datagen.flow_from_directory(
        directory=config['test_image_path'],
        target_size=(224,224), 
        batch_size=32,
        classes=None,
        shuffle=False
    )

    filenames = generator.filenames
    nb_samples = len(filenames)

    test_data_features = model.predict_generator(generator)

    for i in range(0, nb_samples):
        with open(config['labels_file'], 'r') as f:
            LABELS = f.read().splitlines()
        # classify the input image
        results = []
        for idx in range(0, len(test_data_features[i])):
            results.append([test_data_features[i][idx], LABELS[idx]])

        results = sorted(results, key=lambda x: x[0], reverse=True)

        # build the label
        label = results[0][1]
        proba = results[0][0]
        label = "{}: {:.2f}%".format(label, proba * 100)
        print(filenames[i].split(os.path.sep)[-1])
        print(label)

if __name__ == "__main__":
    test_keras_model()

