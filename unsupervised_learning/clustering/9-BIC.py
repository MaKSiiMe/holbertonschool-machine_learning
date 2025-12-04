#!/usr/bin/env python3
"""Find best number of clusters for GMM using BIC"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer, minimum number of clusters (inclusive)
        kmax: positive integer, maximum number of clusters (inclusive)
        iterations: positive integer, max iterations for EM algorithm
        tol: non-negative float, tolerance for EM algorithm
        verbose: boolean, if True EM prints info

    Returns:
        best_k: best value for k based on its BIC
        best_result: tuple containing (pi, m, S) for best k
        l: numpy.ndarray of shape (kmax - kmin + 1) with log likelihoods
        b: numpy.ndarray of shape (kmax - kmin + 1) with BIC values
        or None, None, None, None on failure
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    n, d = X.shape

    # Set default kmax
    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    # Store results
    log_likelihoods = []
    bic_values = []
    results = []

    for k in range(kmin, kmax + 1):
        # Run EM algorithm
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        # Calculate number of parameters p
        # - Priors: k - 1 (they sum to 1)
        # - Means: k * d
        # - Covariances: k * d * (d + 1) / 2 (symmetric matrices)
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)

        # Calculate BIC = p * ln(n) - 2 * l
        bic = p * np.log(n) - 2 * log_l

        log_likelihoods.append(log_l)
        bic_values.append(bic)
        results.append((pi, m, S))

    # Convert to numpy arrays
    likelihoods = np.array(log_likelihoods)
    b = np.array(bic_values)

    # Find best k (minimum BIC)
    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, likelihoods, b
