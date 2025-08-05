#!/usr/bin/env python3
"""Module to concatenate two matrices along a specified axis."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specified axis."""

    return np.concatenate((mat1, mat2), axis=axis)
