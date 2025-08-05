#!/usr/bin/env python3
"""Module to return the transpose of a numpy array."""
import numpy as np


def np_transpose(matrix):
    """Returns the transpose of a numpy array."""

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    return matrix.T
