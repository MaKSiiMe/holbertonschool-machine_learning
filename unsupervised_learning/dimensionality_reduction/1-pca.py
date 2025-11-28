#!/usr/bin/env python3
"""PCA module for dimensionality reduction"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) where n is the number of data points
           and d is the number of dimensions
        ndim: new dimensionality of the transformed X

    Returns:
        T: numpy.ndarray of shape (n, ndim) containing the transformed X
    """
    # Center the data by subtracting the mean
    X_centered = X - np.mean(X, axis=0)

    # Perform SVD decomposition
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Get the weight matrix W (first ndim components)
    W = Vt[:ndim].T

    # Transform the data
    T = X_centered @ W

    return T
