#!/usr/bin/env python3
"""Module for calculating marginal probability"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining the data

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing various hypothetical probabilities
        Pr: 1D numpy.ndarray containing the prior beliefs about P

    Returns:
        The marginal probability of obtaining x and n

    Raises:
        ValueError: if n is not a positive integer
        ValueError: if x is not an integer >= 0
        ValueError: if x > n
        TypeError: if P is not a 1D numpy.ndarray
        TypeError: if Pr is not a numpy.ndarray with the same shape as P
        ValueError: if any value in P or Pr is not in [0, 1]
        ValueError: if Pr does not sum to 1
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

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

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

    likelihood = binomial_coeff * (P ** x) * ((1 - P) ** (n - x))
    intersection = likelihood * Pr
    marginal_prob = np.sum(intersection)

    return marginal_prob
