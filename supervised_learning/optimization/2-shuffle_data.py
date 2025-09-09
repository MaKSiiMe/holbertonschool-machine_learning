#!/usr/bin/env python3
"""Shuffle data module."""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the dataset X and Y."""
    idx = np.random.permutation(X.shape[0])
    return X[idx], Y[idx]
