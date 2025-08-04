#!/usr/bin/env python3
"""Module to transpose a matrix."""


def matrix_transpose(matrix):
    """Transposes a matrix."""
    if not isinstance(matrix, list) or not matrix:
        return []

    transposed = list(zip(*matrix))

    return [list(row) for row in transposed]
