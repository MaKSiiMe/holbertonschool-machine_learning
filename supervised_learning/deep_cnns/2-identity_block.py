#!/usr/bin/env python3
"""Module for building an identity block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning
    for Image Recognition (2015)

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12
            F11: number of filters in the first 1x1 convolution
            F3: number of filters in the 3x3 convolution
            F12: number of filters in the second 1x1 convolution

    Returns:
        activated output of the identity block
    """
    F11, F3, F12 = filters

    # He normal initializer with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # First component: 1x1 convolution
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(bn1)

    # Second component: 3x3 convolution
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(relu1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation('relu')(bn2)

    # Third component: 1x1 convolution
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(relu2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Add skip connection (identity)
    add = K.layers.Add()([bn3, A_prev])

    # Final activation
    output = K.layers.Activation('relu')(add)

    return output
