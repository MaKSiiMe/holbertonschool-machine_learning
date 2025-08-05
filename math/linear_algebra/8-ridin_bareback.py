#!/usr/bin/env python3
"""Module for multiplying two matrices."""


def mat_mul(mat1, mat2):
    """Multiplies two matrices."""
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    if not mat1 or not mat2:
        return None

    if not all(isinstance(row, list) for row in mat1) or \
       not all(isinstance(row, list) for row in mat2):
        return None

    if len(mat1[0]) != len(mat2):
        return None

    transposed_mat2 = list(zip(*mat2))

    result = []
    for row1 in mat1:
        new_row = []
        for col in transposed_mat2:
            new_row.append(sum(a * b for a, b in zip(row1, col)))
        result.append(new_row)

    return result
