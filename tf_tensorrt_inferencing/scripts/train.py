#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # Fine tuning and creating TRT Model
# In this notebook we show how to include TensorRT (TRT) 3.0 in a typical deep learning development workflow. The aim is to show you how you can take advantage of TensorRT to dramatically speed up your inference in a simple and straightforward manner.
# 
# In this example we will see how to fine tune a VGG19 architecture trained on Imagenet to categorize different kinds of flower in 5 classes. After fine tuning we will test the accuracy of the model and save it in a format that is understandeable by TensorRT.
# 
# ## Workflow
# 
# In this notebook we explore the "training" aspect of this problem. For this reason we will need to have tensorflow and keras packages installed in addition to TensorRT 3.0 with python interface, UFF and other modules. Referring to the figure that has been shown at the beginning of the webinar, we will be tackling the "training the model" portion of the slide.
# 
# 
# ### Imports
# In the following we import the necessary python packages for this part of the hands-on session
# 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import Keras Modules
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import keras

# Import Tensorflow Modules
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

import numpy as np
from imutils import paths
import cv2
import shutil
import tarfile
import os
import sys
import time

sys.path.append("scripts")
from conf import config

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# In the two cells above we have imported the python pacakges needed to perform training (fine tuning) In the first cell we have imported Keras with tensorflow backhand. In the second cell we have imported tensorflow and in particular all the routines necessary to save a frozen version of the model. 
# 
# ### Training configuration
# In the following we specify configuration parameters that are needed for training 

class EpochHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epochs = 0
    def on_epoch_end(self, epoch, logs={}):
        self.epochs += 1

# ### Fine-tuning
# In the following we show how to load the VGG19 model and finetune it with images from the flower dataset. Finally, after fine tuning, we will save the model as a frozen tensorflow graph.

def finetune_and_freeze_model():
    # create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')

    # add a global spatial average pooling layer
    x = base_model.output
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a softmax layer -- in this example we have 2 classes
    predictions = Dense(config["output_size"], activation='softmax')(x)

    # this is the model we will finetune
    model = Model(inputs=base_model.input, outputs=predictions)

    # We want to use the convolutional layers from the pretrained
    # VGG19 as feature extractors, so we freeze those layers and exclude
    # them from training and train only the new top layers
    for layer in base_model.layers:
        print(layer.get_config())
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics = ['accuracy']
    )

    if not os.path.exists(config['image_data_dir']):
        print('Extracting images from archive...')
        t = time.time()
        for file in os.listdir(config['images_dir']):
            fn_with_path = os.path.join(config['images_dir'], file)
            if os.path.isfile(fn_with_path) and fn_with_path.endswith('tar.gz'):
                with tarfile.open(fn_with_path) as tar:
                    tar.extractall(path=config['images_dir'])
        print("Extraction done. It took {:.2f} s.".format(time.time() - t))

    if not os.path.exists(config['train_data_dir']):
        print('Splitting dataset...')
        t = time.time()
        if not os.path.exists(config['val_data_dir']):
            os.mkdir(config['val_data_dir'])
        shutil.copytree(config['image_data_dir'], config['train_data_dir'])
        subdirs_created = False
        for root, subdirs, files in os.walk(config['train_data_dir']):
            if not subdirs_created:
                for subdir in sorted(subdirs):
                    os.mkdir(os.path.join(config['val_data_dir'], subdir))
                subdirs_created = True
            sorted_files = sorted(files, reverse=True)
            # partition the data into training and validation splits using 75% of
            # the data for training and the remaining 25% for validation
            for idx in range(0, int(len(sorted_files) * 0.25)):
                os.rename(os.path.join(root, sorted_files[idx]), os.path.join(config['val_data_dir'], root.split(os.path.sep)[-1], sorted_files[idx]))
        print("Split done. It took {:.2f} s.".format(time.time() - t))

    #create data generators for training/validation
    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        directory=config['train_data_dir'],
        target_size=(config['image_dim'], config['image_dim']),
        batch_size=config['train_batch_size']
    )

    config['num_train_samples'] = sum([len(files) for _ , _, files in os.walk(config['train_data_dir'])])

    imagePaths = sorted(list(paths.list_images(config['train_data_dir'])))
    rgb_mean = [0, 0, 0]

    # loop over the training images
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image, 3)
        image = cv2.resize(image, (config['image_dim'], config['image_dim']))
        means = cv2.mean(image)
        rgb_mean[0] += means[0]
        rgb_mean[1] += means[1]
        rgb_mean[2] += means[2]

    print('RGB mean values:')
    print([color / len(imagePaths) for color in rgb_mean])

    val_generator = val_datagen.flow_from_directory(
        directory=config['val_data_dir'],
        target_size=(config['image_dim'], config['image_dim']),
        batch_size=config['train_batch_size']
    )

    config['num_val_samples'] = sum([len(files) for _ , _, files in os.walk(config['val_data_dir'])])

    history = EpochHistory()

    checkpoint = ModelCheckpoint(config["model_file"], monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=1, mode='auto')

    # train the model on the new data for a few epochs
    H = model.fit_generator(
        train_generator,
        steps_per_epoch=config['num_train_samples']//config['train_batch_size'],
        epochs=config['epochs'],
        validation_data=val_generator,
        validation_steps=config['num_val_samples']//config['train_batch_size'],
        callbacks = [checkpoint, early, history]
    )

    #model.save(config["model_file"])

    plt.style.use("ggplot")
    plt.figure()
    #N = config['epochs']
    N = history.epochs # actual number of epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on 3-class Flower-Dataset with VGG19")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config["plot_file"])

    # Now, let's use the Tensorflow backend to get the TF graphdef and frozen graph
    K.set_learning_phase(0)
    sess = K.get_session()
    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

    # save model weights in TF checkpoint
    checkpoint_path = saver.save(sess, config['snapshot_dir'], global_step=0, latest_filename='checkpoint_state')

    # remove nodes not needed for inference from graph def
    train_graph = sess.graph
    inference_graph = tf.graph_util.remove_training_nodes(train_graph.as_graph_def())

    # write the graph definition to a file. 
    # You can view this file to see your network structure and 
    # to determine the names of your network's input/output layers.
    graph_io.write_graph(inference_graph, '.', config['graphdef_file'])

    # specify which layer is the output layer for your graph.
    # In this case, we want to specify the softmax layer after our
    # last dense (fully connected) layer.
    out_names = config['out_layer']

    # freeze your inference graph and save it for later! (Tensorflow)
    freeze_graph.freeze_graph(
        config['graphdef_file'],
        '',
        False,
        checkpoint_path,
        out_names,
        "save/restore_all",
        "save/Const:0",
        config['frozen_model_file'],
        False,
        ""
    )


if __name__ == "__main__":
    # ### Run fine-tuning
    # Fine-tuning will run for the specified number of epochs

    finetune_and_freeze_model()

