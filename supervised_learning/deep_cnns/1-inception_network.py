#!/usr/bin/env python3
"""Module for building the Inception Network (GoogLeNet)"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described in Going Deeper with Convolutions

    Returns:
        The keras model
    """
    # Define input layer with shape (224, 224, 3)
    input_layer = K.Input(shape=(224, 224, 3))

    # Initial convolution and pooling layers
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        activation='relu'
    )(input_layer)

    max_pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(conv1)

    conv2_reduce = K.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )(max_pool1)

    conv2 = K.layers.Conv2D(
        filters=192,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )(conv2_reduce)

    max_pool2 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(conv2)

    # Inception blocks 3a and 3b
    inception_3a = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])

    max_pool3 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(inception_3b)

    # Inception blocks 4a, 4b, 4c, 4d, and 4e
    inception_4a = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])

    max_pool4 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(inception_4e)

    # Inception blocks 5a and 5b
    inception_5a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])

    # Average pooling
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(inception_5b)

    # Dropout layer
    dropout = K.layers.Dropout(rate=0.4)(avg_pool)

    # Fully connected layer with softmax activation
    output_layer = K.layers.Dense(
        units=1000,
        activation='softmax'
    )(dropout)

    # Create the model
    model = K.Model(inputs=input_layer, outputs=output_layer)

    return model
