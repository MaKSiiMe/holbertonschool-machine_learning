#!/usr/bin/env python3
"""Maximization step in the EM algorithm for a GMM"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) containing the posterior
           probabilities for each data point in each cluster

    Returns:
        pi: numpy.ndarray of shape (k,) containing updated priors
        m: numpy.ndarray of shape (k, d) containing updated centroid means
        S: numpy.ndarray of shape (k, d, d) containing updated covariance
           matrices
        or None, None, None on failure
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    # Check dimensions match
    if g.shape[1] != n:
        return None, None, None

    # Check that posterior probabilities sum to 1 for each data point
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    # Calculate N_k (effective number of points in each cluster)
    N_k = np.sum(g, axis=1)

    # Update priors: pi_k = N_k / n
    pi = N_k / n

    # Update means: m_k = (1/N_k) * sum(g_k(i) * x_i)
    # g: (k, n), X: (n, d) -> m: (k, d)
    m = np.dot(g, X) / N_k[:, np.newaxis]

    # Update covariance matrices
    S = np.zeros((k, d, d))

    for i in range(k):
        # X_centered: (n, d)
        X_centered = X - m[i]
        # Weighted covariance
        S[i] = np.dot(g[i] * X_centered.T, X_centered) / N_k[i]

    return pi, m, S
