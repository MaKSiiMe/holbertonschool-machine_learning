#!/usr/bin/env python3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network model using Keras (no Input() allowed)."""
    if len(layers) != len(activations):
        raise ValueError("layers and activations must have same length.")
    if not (0 < keep_prob <= 1):
        raise ValueError("keep_prob must be in (0, 1].")

    model = Sequential()
    drop_rate = 1.0 - keep_prob

    for i, (units, act) in enumerate(zip(layers, activations)):
        model.add(Dense(units=units,
                        activation=act,
                        kernel_regularizer=l2(lambtha)))

        if i < len(layers) - 1 and drop_rate > 0:
            model.add(Dropout(rate=drop_rate))

    model.build((None, nx))
    return model
