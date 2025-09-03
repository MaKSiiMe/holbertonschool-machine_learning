#!/usr/bin/env python3
"""25. One-Hot Decode"""
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


def one_hot_decode(one_hot):
    """Converts a one-hot matrix back into a numeric label vector."""
    if not isinstance(one_hot, np.ndarray):
        return None
    if one_hot.ndim != 2:
        return None
    if one_hot.size == 0:
        return None

    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None

    if not np.all(np.sum(one_hot, axis=0) == 1):
        return None

    return np.argmax(one_hot, axis=0)
