#!/usr/bin/env python3
"""Module to return the shape of a numpy array."""
import numpy as np


def np_shape(matrix):
    """Returns the shape of a numpy array."""

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    return matrix.shape
