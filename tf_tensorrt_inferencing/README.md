TensorFlow -> TensorRT inferencing
===

This is a simple example that does transfer learning with VGG19 on a dataset with 3 classes.

The goal is to compare the prediction results of Keras, TensorRT (Python) and the image classifier example (C++).

As of now all predictions are slightly different which is obviously not good.

**DO NOT TRY THIS AT HOME!** You have been warned ;-)

## Credits

This example is largely based on:

* https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification
* https://github.com/parallel-forall/code-samples/tree/master/posts/TensorRT-3.0
* https://www.pyimagesearch.com/2017/12/18/keras-deep-learning-raspberry-pi/

The training images are a shameless plug from the TensorFlow flower dataset (daisy, dandelion) combined with a subset of the images provided in the PyImageSearch article (not-santa).

## Setup

0. Prerequisites

   Be sure that you have the following libraries installed:
   * CUDA 9.0
   * OpenCV 3.4.1 (including contrib modules and built for Python 3!)
   * cuDNN 7.0.5
   * TensorRT 3.0.4
   * TensorFlow 1.7

1. Clone the repo and build

    ```
    git clone https://github.com/fischermario/deeplearning-playground.git
    cd deeplearning-playground/tf_tensorrt_inferencing
    mkdir build
    cd build
    cmake ..
    make 
    cd ..
    ```

2. Train the model

   Note: all following commands have to be executed in "deeplearning-playground/tf_tensorrt_inferencing"!

    ```
    python3 scripts/train.py
    ```

3. Convert the model and create TensorRT engine file

    ```
    python3 scripts/convert.py
    ```

4. Test the model with Keras, with TensorRT (Python) and with the image classifier example (C++)

    ```
    python3 scripts/test_keras.py
    python3 scripts/test_trt.py
    python3 scripts/test_classifier.py
    ```

    Now compare the results!
