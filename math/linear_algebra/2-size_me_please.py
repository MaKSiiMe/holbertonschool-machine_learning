#!/usr/bin/env python3
"""Module to calculate the shape of a matrix."""


def matrix_shape(matrix):
    """Calculates the shape of a matrix."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if not matrix:
            break
        matrix = matrix[0]

    return shape
