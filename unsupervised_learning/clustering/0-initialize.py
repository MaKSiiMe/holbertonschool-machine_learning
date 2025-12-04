#!/usr/bin/env python3
"""Initialize cluster centroids for K-means"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
           n is the number of data points
           d is the number of dimensions for each data point
        k: positive integer containing the number of clusters

    Returns:
        numpy.ndarray of shape (k, d) containing the initialized centroids,
        or None on failure
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    if k > X.shape[0]:
        return None

    # Get dimensions
    n, d = X.shape

    # Find min and max values along each dimension
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Initialize centroids using uniform distribution
    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    return centroids
