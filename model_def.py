# model_def.py
"""
Model constructors for weights-only .h5 files.

- The VGG16-based classifier architecture below matches the snippet you provided:
    predictions = Dense(1, activation='sigmoid')(x)

- Provide your real resnet_bs() implementation below (replace placeholder)
  if your rib-suppression model is weights-only (saved by model.save_weights()).

Save this file next to app.py and your .h5 weight files.
"""

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# VGG16 classifier input size (from your notebook)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

def build_vgg16_model():
    """
    Build the VGG16-based classifier head you showed in the notebook.
    Returns a Keras Model with output Dense(1, activation='sigmoid').
    """
    # Load pre-trained VGG16 without top layers
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    )

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Classifier constructors expected by app.py
def pneumonia_model():
    return build_vgg16_model()

def nrds_model():
    """
    Your custom NRDS classifier architecture with input 150x150x1.
    """
    model = Sequential()
    model.add(layers.Conv2D(32, (3,3), strides=1, padding='same',
                            activation='relu', input_shape=(150,150,1)))
    model.add(BatchNormalization())
    model.add(layers.MaxPool2D((2,2), strides=2, padding='same'))
    
    model.add(layers.Conv2D(64, (3,3), strides=1, padding='same', activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(BatchNormalization())
    model.add(layers.MaxPool2D((2,2), strides=2, padding='same'))
    
    model.add(layers.Conv2D(64, (3,3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.MaxPool2D((2,2), strides=2, padding='same'))
    
    model.add(layers.Conv2D(128, (3,3), strides=1, padding='same', activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(BatchNormalization())
    model.add(layers.MaxPool2D((2,2), strides=2, padding='same'))
    
    model.add(layers.Conv2D(256, (3,3), strides=1, padding='same', activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(BatchNormalization())
    model.add(layers.MaxPool2D((2,2), strides=2, padding='same'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model
def covid19_model():
    return build_vgg16_model()

def lung_opacity_model():
    return build_vgg16_model()

# -------------------------
# Placeholder resnet_bs()
# -------------------------

def res_block(x_in, filters, scaling):
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x_in)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x
def resnet_bs(num_filters=64, num_res_blocks=16, res_block_scaling=0.1):
    x_in = Input(shape=(256, 256, 1))
    x = b = Conv2D(num_filters, (3, 3), padding='same')(x_in)
    
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)

    b = Conv2D(num_filters, (3, 3), padding='same')(b)
    x = Add()([x, b])
    x = Conv2D(1, (3, 3), padding='same')(x)

    return Model(x_in, x, name="ResNet-BS")
