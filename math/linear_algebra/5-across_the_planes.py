#!/usr/bin/env python3
"""Module to add two 2D matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise."""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[x + y for x, y in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
