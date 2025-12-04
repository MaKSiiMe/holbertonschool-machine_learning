#!/usr/bin/env python3
"""Initialize variables for a Gaussian Mixture Model"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters

    Returns:
        pi: numpy.ndarray of shape (k,) containing priors for each cluster
        m: numpy.ndarray of shape (k, d) containing centroid means
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices
        or None, None, None on failure
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    if k > n:
        return None, None, None

    # Initialize priors evenly
    pi = np.full((k,), 1 / k)

    # Initialize centroids with K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # Initialize covariance matrices as identity matrices
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
