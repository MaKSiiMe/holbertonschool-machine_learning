#!/usr/bin/env python3
"""Module to transpose a matrix."""


def matrix_transpose(matrix):
    """Transposes a matrix."""
    transposed = list(zip(*matrix))

    return [list(row) for row in transposed]
