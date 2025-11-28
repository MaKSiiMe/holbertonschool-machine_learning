#!/usr/bin/env python3
"""PCA module for dimensionality reduction"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) where n is the number of data points
           and d is the number of dimensions. All dimensions have mean of 0.
        var: fraction of variance that PCA transformation should maintain

    Returns:
        W: numpy.ndarray of shape (d, nd) - weights matrix that maintains
           var fraction of X's original variance
    """
    # Perform SVD decomposition
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Calculate variance explained by each component
    # Variance is proportional to squared singular values
    variance = S ** 2

    # Calculate cumulative variance ratio
    total_variance = np.sum(variance)
    cumulative_variance_ratio = np.cumsum(variance) / total_variance

    # Find number of components needed to reach desired variance
    nd = np.argmax(cumulative_variance_ratio >= var) + 1

    # Return the weight matrix W (first nd columns of V^T transposed)
    W = Vt[:nd].T

    return W
