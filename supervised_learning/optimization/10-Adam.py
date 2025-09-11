#!/usr/bin/env python3
"""TensorFlow Adam optimizer setup."""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Sets up the Adam optimization algorithm in TensorFlow."""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )

    return optimizer
