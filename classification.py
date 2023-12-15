#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sklearn
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.applications.inception_v3
import tensorflow.keras.layers as tfl
from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from livelossplot.inputs.tf_keras import PlotLossesCallback
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from platform import python_version
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from pathlib import Path
from tensorflow.keras.models import model_from_json
from sklearn.datasets import load_svmlight_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from tensorflow.keras.applications import InceptionV3
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

ImageFile.LOAD_TRUNCATED_IMAGES = True

print(python_version())
print(tf.__version__)
print(tf.keras.__version__)
print(mp.__version__)

global weights_file
weights_file = "model_weights/global_mobilenet_pv_pv_white.h5"
global model_file
model_file = "model_weights/global_mobilenet_pv_pv_white.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 256
SHAPE = (224,224,3)
global inference_model
inference_model=None


def predict_disease(test_directory):

    global inference_model

    if(inference_model==None):
        print("Initializing inference model...")
        # load json and create model
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        inference_model = model_from_json(loaded_model_json)
        # load weights into new model
        inference_model.load_weights(weights_file)
        print("Loaded model from disk")

    predict_datagen = ImageDataGenerator()
    print("test_directory", test_directory)
    if os.path.isdir(test_directory):
        test_generator = predict_datagen.flow_from_directory(
                    #base_path + 'test/',
                    test_directory,
                    target_size = (SHAPE[0], SHAPE[1]),
                    batch_size = 128,
                    class_mode = 'categorical',
                    shuffle = False,#Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.
                    #seed = 33,
                    #classes = test_folder_names
        )
    else:
        img = load_img(test_directory, target_size=(SHAPE[0], SHAPE[1]))
        test_generator = img_to_array(img)

    # ### COMPUTE PREDICTIONS ON TEST DATA ###
    class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry___healthy',
                   'Corn___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
                   'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                   'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                   'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                   'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
                   'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

    pred_test = inference_model.predict(test_generator)
    #print("raw prediction without max = ", type(pred_test), pred_test)
    pred_test = np.argmax(pred_test, axis=1)

    #print("raw prediction = ", type(pred_test), pred_test)
    #print("test_generator.filenames = ", type(test_generator.filenames), test_generator.filenames)
    pred_test = [class_labels[i] for i in pred_test.tolist()]
    #print("predicted labels = ", type(pred_test), pred_test)

    return pred_test
