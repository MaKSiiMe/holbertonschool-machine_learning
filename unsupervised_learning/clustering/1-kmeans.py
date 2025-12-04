#!/usr/bin/env python3
"""K-means clustering algorithm"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer containing max iterations

    Returns:
        C: numpy.ndarray of shape (k, d) with centroid means for each cluster
        clss: numpy.ndarray of shape (n,) with cluster index for each point
        or None, None on failure
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    if k > n:
        return None, None

    # Initialize centroids using uniform distribution
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    # K-means iterations
    for _ in range(iterations):
        # Save previous centroids for convergence check
        C_prev = np.copy(C)

        # Calculate distances from each point to each centroid
        # Using broadcasting: (n, 1, d) - (k, d) -> (n, k, d) -> sum -> (n, k)
        distances = np.sqrt(np.sum((X[:, np.newaxis] - C) ** 2, axis=2))

        # Assign each point to nearest centroid
        clss = np.argmin(distances, axis=1)

        # Update centroids
        for i in range(k):
            # Get points assigned to cluster i
            mask = clss == i

            if np.sum(mask) == 0:
                # Reinitialize empty cluster centroid
                C[i] = np.random.uniform(low=min_vals, high=max_vals,
                                         size=(d,))
            else:
                # Update centroid as mean of assigned points
                C[i] = np.mean(X[mask], axis=0)

        # Check for convergence
        if np.array_equal(C, C_prev):
            break

    # Final assignment after last update
    distances = np.sqrt(np.sum((X[:, np.newaxis] - C) ** 2, axis=2))
    clss = np.argmin(distances, axis=1)

    return C, clss
