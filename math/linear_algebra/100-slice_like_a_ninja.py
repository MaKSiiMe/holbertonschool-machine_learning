#!/usr/bin/env python3
"""Module to slice a numpy array like a ninja."""


def np_slice(matrix, axes={}):
    """Slices a numpy array according to the axes provided."""

    slices = tuple(slice(*axes.get(i, (None, None, None)))
                   for i in range(matrix.ndim))

    return matrix[slices]
