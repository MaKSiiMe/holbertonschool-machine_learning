#!/usr/bin/env python3
"""TensorFlow RMSProp optimizer setup."""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Sets up the RMSProp optimization algorithm in TensorFlow."""
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )

    return optimizer
