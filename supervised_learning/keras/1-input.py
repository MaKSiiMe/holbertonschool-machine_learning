#!/usr/bin/env python3
"""Module for building neural network with Keras functional API."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network model using Keras functional API."""
    if len(layers) != len(activations):
        raise ValueError("layers and activations must have same length.")
    if not (0 < keep_prob <= 1):
        raise ValueError("keep_prob must be in (0, 1].")

    drop_rate = 1.0 - keep_prob

    inputs = K.Input(shape=(nx,))
    x = inputs

    for i, (units, act) in enumerate(zip(layers, activations)):
        x = K.layers.Dense(units=units,
                           activation=act,
                           kernel_regularizer=K.regularizers.l2(lambtha))(x)

        if i < len(layers) - 1 and drop_rate > 0:
            x = K.layers.Dropout(rate=drop_rate)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
