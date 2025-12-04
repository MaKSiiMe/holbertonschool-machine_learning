#!/usr/bin/env python3
"""Initialize cluster centroids with K-means"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids with K-means.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
           n is the number of data points
           d is the number of dimensions in each data point
        k: positive integer containing the number of clusters

    Returns:
        numpy.ndarray of shape (k, d) containing the initialized centroids,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    if k > X.shape[0]:
        return None

    n, d = X.shape

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    return centroids
