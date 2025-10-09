#!/usr/bin/env python3
"""Module for building an Inception block"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going Deeper with Convolutions

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F1, F3R, F3, F5R, F5, FPP
            F1: number of filters in the 1x1 convolution
            F3R: number of filters in the 1x1 convolution before 3x3 conv
            F3: number of filters in the 3x3 convolution
            F5R: number of filters in the 1x1 convolution before 5x5 conv
            F5: number of filters in the 5x5 convolution
            FPP: number of filters in the 1x1 convolution after max pooling

    Returns:
        Concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Branch 1: 1x1 convolution
    conv1x1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    # Branch 2: 1x1 convolution -> 3x3 convolution
    conv3x3_reduce = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)
    conv3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(conv3x3_reduce)

    # Branch 3: 1x1 convolution -> 5x5 convolution
    conv5x5_reduce = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)
    conv5x5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )(conv5x5_reduce)

    # Branch 4: Max pooling -> 1x1 convolution
    max_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)
    pool_proj = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(max_pool)

    # Concatenate all branches along the channel axis
    output = K.layers.concatenate([conv1x1, conv3x3, conv5x5, pool_proj])

    return output
