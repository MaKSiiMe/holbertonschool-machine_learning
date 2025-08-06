#!/usr/bin/env python3
"""Module to add two matrices element-wise."""


def add_matrices(mat1, mat2):
    """Adds two matrices element-wise."""

    if not isinstance(mat1, list):
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        element_sum = add_matrices(mat1[i], mat2[i])
        if element_sum is None:
            return None
        result.append(element_sum)

    return result
