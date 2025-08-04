#!/usr/bin/env python3

def matrix_shape(matrix):
    """Calculates the shape of a matrix."""
    if not isinstance(matrix, list) or not matrix:
        return []

    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if not matrix:
            break
        matrix = matrix[0]

    return shape
