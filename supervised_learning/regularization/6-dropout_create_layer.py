#!/usr/bin/env python3
"""6. Create a Layer with Dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a layer of a neural network using dropout"""
    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_avg')
    )

    layer_output = dense_layer(prev)

    dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)

    return dropout_layer(layer_output, training=training)
