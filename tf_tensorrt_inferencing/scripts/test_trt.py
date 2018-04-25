#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # Predictions with Tensor RT
# In this notebook we show how to use the Tensor RT engine that we created after training our model and serialized to disk. 
# 
# ### Imports
# We import here the packages that are required to run this notebook

from tensorrt.lite import Engine

from PIL import Image
import numpy as np
import os
import functools
import time

import matplotlib.pyplot as plt

import sys
sys.path.append("scripts")
from conf import config


# ### Utility functions
# We define here a few utility functions. These functions are used to 
# * Analyze the prediction
# * Convert image to a format that is identical to the format used durign training
# * Organize the images into a list of numpy array
# * Time the compute time of a function

def analyze(output_data):
    with open(config['labels_file'], 'r') as f:
        LABELS = f.read().splitlines()
    output = output_data.reshape(-1, len(LABELS))
    
    top_classes = [LABELS[idx] for idx in np.argmax(output, axis=1)]
    top_classes_prob = np.amax(output, axis=1)  

    return top_classes, top_classes_prob


def image_to_np_CHW(image):
    #print(image.filename)
    return np.asarray(
        image.resize(
            (224, 224), 
            Image.ANTIALIAS
        )).transpose([2,0,1]).astype(np.float32)


def load_and_preprocess_images():
    images_trt = []
    images_filenames = []
    for root, _, files in os.walk(config["test_image_path"]):
        for filename in sorted(files):
            fn_with_path = os.path.join(root, filename)
            if os.path.isfile(fn_with_path):
                images_trt.append(image_to_np_CHW(Image.open(fn_with_path)))
                images_filenames.append(filename)
        
    images_trt = np.stack(images_trt)
    
    num_batches = int(len(images_trt) / config["inference_batch_size"])
    
    images_trt = np.reshape(images_trt[0:num_batches * config["inference_batch_size"]], [
        num_batches, 
        config["inference_batch_size"], 
        images_trt.shape[1],
        images_trt.shape[2],
        images_trt.shape[3]
    ])

    return [images_filenames, images_trt]


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        retargs = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return retargs
    return newfunc

@timeit
def infer_all_images_trt(engine, images_trt, images_filenames):
    results = []
    for idx in range(0, len(images_trt)):
        result = engine.infer(images_trt[idx])
        results.append([result, images_trt[idx], images_filenames[idx]])
    return results

def predict_all_images():
    # ### Prepare TensorRT Engine
    # Here we simply load the TRT engine such that we can do inference. We can also attach a function (utility function) to postprocess the outputs before returning them (in this case we use the function analyze)

    plan = str(config["engine_save_dir"]) + "keras_vgg19_b" + str(config["inference_batch_size"]) + "_"+ str(config["precision"]) + ".engine"

    engine = Engine(PLAN=plan, postprocessors={"dense_2/Softmax":analyze})

    # ### Load all data
    # Here we load all the test data from the directory specified above in "test_image_path"

    (images_filenames, images_trt) = load_and_preprocess_images()

    # ### Prepare function to do inference with Tensor RT

    results_trt = infer_all_images_trt(engine, images_trt, images_filenames)

    for i in range(len(results_trt)):
        print(results_trt[i][2])
        label = "{}: {:.2f}%".format(results_trt[i][0][0][0][0][0], float(results_trt[i][0][0][0][1][0]) * 100)
        print(label)
        #plt.imshow(results_trt[i][1][0, 0],  cmap='gray')
        #plt.show()

if __name__ == "__main__":
    predict_all_images()

