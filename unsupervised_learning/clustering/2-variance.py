#!/usr/bin/env python3
"""Calculate total intra-cluster variance"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        C: numpy.ndarray of shape (k, d) containing centroid means

    Returns:
        var: total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    squared_distances = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)
    min_squared_distances = np.min(squared_distances, axis=1)
    var = np.sum(min_squared_distances)

    return var
