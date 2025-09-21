#!/usr/bin/env python3
"""3. Create a Layer with L2 Regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a neural network layer in TensorFlow
    that includes L2 regularization"""
    l2_regularizer = tf.keras.regularizers.l2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_avg'),
        kernel_regularizer=l2_regularizer
    )

    return layer(prev)
