#!/usr/bin/env python3
"""24. One-Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix."""
    if not isinstance(Y, np.ndarray):
        return None
    if Y.ndim != 1:
        return None
    if not isinstance(classes, int):
        return None
    if classes <= 0:
        return None
    if Y.size == 0:
        return None
    if np.any(Y >= classes) or np.any(Y < 0):
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1

    return one_hot
