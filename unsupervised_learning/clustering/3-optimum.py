#!/usr/bin/env python3
"""Find optimum number of clusters by variance"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer, minimum number of clusters to check
              (inclusive)
        kmax: positive integer, maximum number of clusters to check
              (inclusive)
        iterations: positive integer, max iterations for K-means

    Returns:
        results: list containing outputs of K-means for each cluster size
        d_vars: list containing difference in variance from smallest
                cluster size
        or None, None on failure
    """
    # Validate X
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    n, d = X.shape

    # Set default kmax
    if kmax is None:
        kmax = n

    # Validate kmin, kmax, iterations
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []

    # Run K-means for each k value
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        variances.append(variance(X, C))

    # Calculate difference in variance from smallest cluster size (kmin)
    var_kmin = variances[0]
    d_vars = [var_kmin - var for var in variances]

    return results, d_vars
