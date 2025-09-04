#!/usr/bin/env python3
"""Module for optimizing a Keras model using Adam optimizer."""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for a Keras model
    with categorical crossentropy loss and accuracy metrics."""
    adam = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
