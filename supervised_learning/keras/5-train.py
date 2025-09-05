#!/usr/bin/env python3
"""Module for training a Keras model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, verbose=True, shuffle=False):
    """Trains a Keras model using mini-batch gradient descent."""
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
