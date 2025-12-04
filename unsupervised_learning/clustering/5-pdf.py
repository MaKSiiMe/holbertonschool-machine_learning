#!/usr/bin/env python3
"""Calculate PDF of a Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing data points
        m: numpy.ndarray of shape (d,) containing the mean of the
           distribution
        S: numpy.ndarray of shape (d, d) containing the covariance matrix

    Returns:
        P: numpy.ndarray of shape (n,) containing PDF values for each data point
        or None on failure
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    n, d = X.shape

    # Check dimensions match
    if m.shape[0] != d:
        return None
    if S.shape[0] != d or S.shape[1] != d:
        return None

    # Calculate determinant of S
    det_S = np.linalg.det(S)

    # Calculate inverse of S
    S_inv = np.linalg.inv(S)

    # Calculate normalization constant
    # (2π)^(d/2) × |S|^(1/2)
    norm_const = np.sqrt(((2 * np.pi) ** d) * det_S)

    # Calculate (X - m)
    X_centered = X - m

    # Calculate the exponent: -0.5 × (X-m)ᵀ S⁻¹ (X-m)
    # For each point: (X-m) @ S_inv @ (X-m).T
    # Using einsum for efficient computation without loops
    exponent = -0.5 * np.sum(X_centered @ S_inv * X_centered, axis=1)

    # Calculate PDF
    P = np.exp(exponent) / norm_const

    # Set minimum value to 1e-300
    P = np.maximum(P, 1e-300)

    return P
