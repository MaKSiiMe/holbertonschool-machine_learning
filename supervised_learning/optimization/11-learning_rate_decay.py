#!/usr/bin/env python3
"""Learning rate decay module."""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy."""
    decay_periods = np.floor(global_step / decay_step)

    updated_alpha = alpha / (1 + decay_rate * decay_periods)

    return updated_alpha
