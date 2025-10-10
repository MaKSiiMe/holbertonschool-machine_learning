#!/usr/bin/env python3
"""Builds the DenseNet-121 architecture."""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds DenseNet-121 (Huang et al.) with BN → ReLU → Conv ordering.
    - growth_rate: k
    - compression: θ
    Returns: keras.Model
    """
    he_init = K.initializers.he_normal(seed=0)

    # Input
    X_input = K.Input(shape=(224, 224, 3))

    # Stem: BN → ReLU → Conv7x7 s=2 → MaxPool3x3 s=2
    X = K.layers.BatchNormalization(axis=3)(X_input)
    X = K.layers.ReLU()(X)
    X = K.layers.Conv2D(
        2 * growth_rate, (7, 7), strides=(2, 2),
        padding='same', kernel_initializer=he_init
    )(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

    nb_filters = 2 * growth_rate  # initial number of filters

    # Dense Blocks + Transition layers: (6, 12, 24, 16)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)  # no transition after

    # Head: BN → ReLU → GlobalAvgPool → Dense(1000, softmax)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.ReLU()(X)
    X = K.layers.GlobalAveragePooling2D()(X)
    outputs = K.layers.Dense(1000, activation='softmax', kernel_initializer=he_init)(X)

    return K.Model(inputs=X_input, outputs=outputs)
