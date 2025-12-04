#!/usr/bin/env python3
"""Calculate PDF of a Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing data points
        m: numpy.ndarray of shape (d,) containing the mean
        S: numpy.ndarray of shape (d, d) containing the covariance matrix

    Returns:
        P: numpy.ndarray of shape (n,) containing PDF values
        or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d:
        return None
    if S.shape[0] != d or S.shape[1] != d:
        return None

    det_S = np.linalg.det(S)
    S_inv = np.linalg.inv(S)

    norm_const = np.sqrt(((2 * np.pi) ** d) * det_S)

    X_centered = X - m

    exponent = -0.5 * np.sum(X_centered @ S_inv * X_centered, axis=1)

    P = np.exp(exponent) / norm_const

    P = np.maximum(P, 1e-300)

    return P
