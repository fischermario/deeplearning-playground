#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ### Create TRT engine
# 
# In this notebook we create and save a TensorRT engine that we can use in deployment. Be sure that your GPU memory has been freed before running this part of the code.

# ### Imports
# 
# In this block we import the necessary python packages

# Import TensorRT Modules
import tensorrt as trt
import uff
from tensorrt.parsers import uffparser

import os


# ### Configuration

import sys
sys.path.append("scripts")
from conf import config


# ## UFF Model for TRT
# 
# In the following function we load the frozen model (tensorflow) and we create the UFF engine. In order to create it we need to sspecify (among other things) the precision. Note that the function call for half and full precision differs only by one argument (

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)


def create_and_save_inference_engine():
    # Define network parameters, including inference batch size, name & dimensionality of input/output layers
    INPUT_LAYERS = [config['input_layer']]
    OUTPUT_LAYERS = [config['out_layer']]
    INFERENCE_BATCH_SIZE = config['inference_batch_size']

    INPUT_C = 3
    INPUT_H = config['image_dim']
    INPUT_W = config['image_dim']

    # Load your newly created Tensorflow frozen model and convert it to UFF
    uff_model = uff.from_tensorflow_frozen_model(config['frozen_model_file'], OUTPUT_LAYERS)

    # Create a UFF parser to parse the UFF file created from your TF Frozen model
    parser = uffparser.create_uff_parser()
    parser.register_input(INPUT_LAYERS[0], (INPUT_C,INPUT_H,INPUT_W),0)
    parser.register_output(OUTPUT_LAYERS[0])

    # Build your TensorRT inference engine
    if(config['precision'] == 'fp32'):
        engine = trt.utils.uff_to_trt_engine(
            G_LOGGER, 
            uff_model, 
            parser, 
            INFERENCE_BATCH_SIZE, 
            1<<20, 
            trt.infer.DataType.FLOAT
        )

    elif(config['precision'] == 'fp16'):
        engine = trt.utils.uff_to_trt_engine(
            G_LOGGER, 
            uff_model, 
            parser, 
            INFERENCE_BATCH_SIZE, 
            1<<20, 
            trt.infer.DataType.HALF
        )
    
    # Serialize TensorRT engine to a file for when you are ready to deploy your model.
    save_path = str(config['engine_save_dir']) + "keras_vgg19_b" + str(INFERENCE_BATCH_SIZE) + "_"+ str(config['precision']) + ".engine"

    trt.utils.write_engine_to_file(save_path, engine.serialize())
    
    print("Saved TRT engine to {}".format(save_path))


if __name__ == "__main__":
    create_and_save_inference_engine()

