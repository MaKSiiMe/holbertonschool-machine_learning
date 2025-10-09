#!/usr/bin/env python3
"""Module for building DenseNet-121 architecture"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks

    Args:
        growth_rate: growth rate for the dense blocks
        compression: compression factor for transition layers

    Returns:
        the keras model
    """
    # He normal initializer with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # Input layer
    X_input = K.Input(shape=(224, 224, 3))

    # Initial layers: BN -> ReLU -> Conv 7x7 stride 2
    X = K.layers.BatchNormalization(axis=3)(X_input)
    X = K.layers.ReLU()(X)
    X = K.layers.Conv2D(
        filters=2 * growth_rate,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(X)

    # Max pooling: 3x3 stride 2
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    # Initial number of filters
    nb_filters = 2 * growth_rate

    # Dense Block 1: 6 layers
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)
    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2: 12 layers
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3: 24 layers
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4: 16 layers (no transition after)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Final layers: BN -> ReLU -> Global Average Pooling
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.ReLU()(X)
    X = K.layers.GlobalAveragePooling2D()(X)

    # Output layer: fully connected with 1000 classes
    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(X)

    # Create model
    model = K.Model(inputs=X_input, outputs=X)

    return model
