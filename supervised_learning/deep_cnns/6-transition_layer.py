#!/usr/bin/env python3
"""Module for building a transition layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected
    Convolutional Networks

    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        compression: compression factor for the transition layer

    Returns:
        The output of the transition layer and the number of filters
        within the output
    """
    # He normal initializer with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # Calculate number of filters after compression
    nb_filters = int(nb_filters * compression)

    # Batch Normalization
    bn = K.layers.BatchNormalization(axis=3)(X)

    # ReLU activation
    relu = K.layers.Activation('relu')(bn)

    # 1x1 Convolution with compression
    conv = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(relu)

    # Average Pooling 2x2 with stride 2
    pool = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same'
    )(conv)

    return pool, nb_filters
