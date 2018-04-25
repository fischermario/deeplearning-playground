import os

config = {
    # Training params
    "images_dir": os.path.join(os.getcwd(), "data/images"),  # images directory
    "image_data_dir": os.path.join(os.getcwd(), "data/images/full_data"),  # all data
    "train_data_dir": os.path.join(os.getcwd(), "data/images/train"),  # training data
    "val_data_dir": os.path.join(os.getcwd(), "data/images/val"),  # validation data 
    "train_batch_size": 16,  # training batch size
    "epochs": 50,  # number of training epochs
    "num_train_samples": 0,  # number of training examples
    "num_val_samples": 0,  # number of test examples

    # Where to save models (Tensorflow + TensorRT)
    "graphdef_file": os.path.join(os.getcwd(), "data/frozen_graph/keras_vgg19_graphdef.pb"),
    "frozen_model_file": os.path.join(os.getcwd(), "data/frozen_graph/keras_vgg19_frozen_model.pb"),
    "model_file": os.path.join(os.getcwd(), "data/model/keras_vgg19.h5"),
    "snapshot_dir": os.path.join(os.getcwd(), "data/snapshot/0"),
    "engine_save_dir": os.path.join(os.getcwd(), "data/engine/"),
    "labels_file": os.path.join(os.getcwd(), "data/keras_vgg19_flowers_labels.txt"),
    
    # Needed for TensorRT
    "image_dim": 224,  # the image size (square images)
    "inference_batch_size": 1,  # inference batch size
    "input_layer": "input_1",  # name of the input tensor in the TF computational graph
    "out_layer": "dense_2/Softmax",  # name of the output tensorf in the TF conputational graph
    "output_size": 3,  # number of classes in output
    "precision": "fp32",  # desired precision (fp32, fp16)

    "test_image_path": os.path.join(os.getcwd(), "data/examples"),
    "plot_file": os.path.join(os.getcwd(), "data/plot/plot.png")
}
