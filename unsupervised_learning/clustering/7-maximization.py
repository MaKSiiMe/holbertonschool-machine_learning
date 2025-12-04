#!/usr/bin/env python3
"""Maximization step in the EM algorithm with a GMM"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm with a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) containing posterior probabilities

    Returns:
        pi: numpy.ndarray of shape (k,) containing updated priors
        m: numpy.ndarray of shape (k, d) containing updated centroid means
        S: numpy.ndarray of shape (k, d, d) containing updated covariance
           matrices
        or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    N_k = np.sum(g, axis=1)

    pi = N_k / n

    m = np.dot(g, X) / N_k[:, np.newaxis]

    S = np.zeros((k, d, d))

    for i in range(k):
        X_centered = X - m[i]
        S[i] = np.dot(g[i] * X_centered.T, X_centered) / N_k[i]

    return pi, m, S
