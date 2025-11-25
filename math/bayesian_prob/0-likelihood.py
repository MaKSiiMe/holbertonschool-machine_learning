#!/usr/bin/env python3
"""Module for calculating likelihood"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data given various hypothetical
    probabilities of developing severe side effects

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing various hypothetical probabilities

    Returns:
        1D numpy.ndarray containing the likelihood for each probability in P

    Raises:
        ValueError: if n is not a positive integer
        ValueError: if x is not an integer >= 0
        ValueError: if x > n
        TypeError: if P is not a 1D numpy.ndarray
        ValueError: if any value in P is not in [0, 1]
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    n_fact = 1
    for i in range(1, n + 1):
        n_fact *= i

    x_fact = 1
    for i in range(1, x + 1):
        x_fact *= i

    nx_fact = 1
    for i in range(1, n - x + 1):
        nx_fact *= i

    binomial_coeff = n_fact / (x_fact * nx_fact)

    likelihood_values = binomial_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihood_values
