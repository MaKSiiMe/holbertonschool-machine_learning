#!/usr/bin/env python3
"""Expectation step in the EM algorithm for a GMM"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        pi: numpy.ndarray of shape (k,) containing priors for each cluster
        m: numpy.ndarray of shape (k, d) containing centroid means
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices

    Returns:
        g: numpy.ndarray of shape (k, n) containing posterior probabilities
        l: total log likelihood
        or None, None on failure
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    # Check dimensions match
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    # Calculate weighted likelihoods for each cluster
    # g will store π_k × PDF_k for each cluster
    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        g[i] = pi[i] * P

    # Sum of weighted likelihoods for each data point
    total_likelihood = np.sum(g, axis=0)

    # Calculate log likelihood
    log_likelihood = np.sum(np.log(total_likelihood))

    # Calculate posterior probabilities (responsibilities)
    g = g / total_likelihood

    return g, log_likelihood
