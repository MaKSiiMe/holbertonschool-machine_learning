#!/usr/bin/env python3
"""Learning rate decay operation in TensorFlow."""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Creates a learning rate decay operation in tensorflow
    using inverse time decay."""
    learning_rate_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    return learning_rate_schedule
