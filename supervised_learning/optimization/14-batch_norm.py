#!/usr/bin/env python3
"""Batch normalization layer creation in TensorFlow."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer
    for a neural network in tensorflow."""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    z = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)

    mean, var = tf.nn.moments(z, axes=0)

    gamma = tf.Variable(
        initial_value=tf.ones([n], dtype=z.dtype),
        trainable=True,
        name="gamma"
    )
    beta = tf.Variable(
        initial_value=tf.zeros([n], dtype=z.dtype),
        trainable=True,
        name="beta"
    )

    eps = 1e-7
    z_norm = (z - mean) / tf.sqrt(var + eps)
    out = gamma * z_norm + beta

    return activation(out)