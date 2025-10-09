#!/usr/bin/env python3
"""Module for building a dense block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected
    Convolutional Networks

    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block

    Returns:
        The concatenated output of each layer within the Dense Block
        and the number of filters within the concatenated outputs
    """
    # He normal initializer with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # Start with the input
    concat_features = X

    for i in range(layers):
        # Bottleneck layer: BN -> ReLU -> Conv 1x1
        bn1 = K.layers.BatchNormalization(axis=3)(concat_features)
        relu1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=initializer
        )(relu1)

        # Composite layer: BN -> ReLU -> Conv 3x3
        bn2 = K.layers.BatchNormalization(axis=3)(conv1)
        relu2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=initializer
        )(relu2)

        # Concatenate with previous features
        concat_features = K.layers.Concatenate(
            axis=3)([concat_features, conv2])

        # Update number of filters
        nb_filters += growth_rate

    return concat_features, nb_filters
