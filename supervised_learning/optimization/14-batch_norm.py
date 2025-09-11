#!/usr/bin/env python3
"""Batch normalization layer creation in TensorFlow."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer
    for a neural network in tensorflow."""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)

    norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )(dense)

    return activation(norm)
