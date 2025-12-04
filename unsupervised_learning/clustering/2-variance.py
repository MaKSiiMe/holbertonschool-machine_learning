#!/usr/bin/env python3
"""Calculate total intra-cluster variance"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        C: numpy.ndarray of shape (k, d) containing centroid means for each cluster

    Returns:
        var: total variance, or None on failure
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Calculate squared distances from each point to each centroid
    # (n, 1, d) - (k, d) -> (n, k, d) -> sum over d -> (n, k)
    squared_distances = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)

    # Find minimum squared distance for each point (to nearest centroid)
    min_squared_distances = np.min(squared_distances, axis=1)

    # Total variance is sum of all minimum squared distances
    var = np.sum(min_squared_distances)

    return var
