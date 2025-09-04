#!/usr/bin/env python3
"""Module for building neural network with Keras Sequential model."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network model using Keras Sequential model."""
    if len(layers) != len(activations):
        raise ValueError("layers and activations must have same length.")
    if not (0 < keep_prob <= 1):
        raise ValueError("keep_prob must be in (0, 1].")

    model = K.Sequential()
    drop_rate = 1.0 - keep_prob

    for i, (units, act) in enumerate(zip(layers, activations)):
        if i == 0:
            model.add(
                K.layers.Dense(units=units,
                               activation=act,
                               kernel_regularizer=K.regularizers.l2(lambtha),
                               input_shape=(nx,))
            )
        else:
            model.add(
                K.layers.Dense(units=units,
                               activation=act,
                               kernel_regularizer=K.regularizers.l2(lambtha))
            )

        if i < len(layers) - 1 and drop_rate > 0:
            model.add(K.layers.Dropout(rate=drop_rate))

    return model
