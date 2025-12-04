#!/usr/bin/env python3
"""Expectation Maximization algorithm with a GMM"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Runs the expectation maximization with a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
        iterations: positive integer, max iterations
        tol: non-negative float, tolerance (early stopping)
        verbose: boolean, if True print log likelihood

    Returns:
        pi: numpy.ndarray of shape (k,) containing priors per cluster
        m: numpy.ndarray of shape (k, d) containing centroid means
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices
        g: numpy.ndarray of shape (k, n) containing probabilities
        l: log likelihood of the model
        or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, log_likelihood = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None

    prev_log_likelihood = None

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(log_likelihood, 5)))

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, log_likelihood = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if prev_log_likelihood is not None:
            if abs(log_likelihood - prev_log_likelihood) <= tol:
                if verbose:
                    print("Log Likelihood after {} iterations: {}".format(
                        i + 1, round(log_likelihood, 5)))
                break

        prev_log_likelihood = log_likelihood
    else:
        if verbose:
            print("Log Likelihood after {} iterations: {}".format(
                iterations, round(log_likelihood, 5)))

    return pi, m, S, g, log_likelihood
