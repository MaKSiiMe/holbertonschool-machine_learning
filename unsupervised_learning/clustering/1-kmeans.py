#!/usr/bin/env python3
"""K-means clustering algorithm"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Runs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer containing max iterations

    Returns:
        C: numpy.ndarray of shape (k, d) with centroid means
        clss: numpy.ndarray of shape (n,) with cluster index per point
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    if k > n:
        return None, None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    for _ in range(iterations):
        C_prev = np.copy(C)

        distances = np.sqrt(np.sum((X[:, np.newaxis] - C) ** 2, axis=2))
        clss = np.argmin(distances, axis=1)

        for i in range(k):
            mask = clss == i

            if np.sum(mask) == 0:
                C[i] = np.random.uniform(low=min_vals, high=max_vals,
                                         size=(d,))
            else:
                C[i] = np.mean(X[mask], axis=0)

        if np.array_equal(C, C_prev):
            break

    distances = np.sqrt(np.sum((X[:, np.newaxis] - C) ** 2, axis=2))
    clss = np.argmin(distances, axis=1)

    return C, clss
