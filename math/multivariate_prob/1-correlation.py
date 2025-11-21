#!/usr/bin/env python3
"""Module for calculating correlation matrix"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix

    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix
           d is the number of dimensions

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix

    Raises:
        TypeError: if C is not a numpy.ndarray
        ValueError: if C is not a 2D square matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")

    d1, d2 = C.shape
    if d1 != d2:
        raise ValueError("C must be a 2D square matrix")

    std = np.sqrt(np.diag(C))
    std_outer = np.outer(std, std)
    correlation_matrix = C / std_outer

    return correlation_matrix
