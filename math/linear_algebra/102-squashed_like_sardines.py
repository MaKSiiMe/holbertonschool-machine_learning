#!/usr/bin/env python3


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate two matrices along a specified axis."""

    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    if axis == 0:
        return mat1 + mat2

    if len(mat1) == 0 or len(mat2) == 0:
        return mat1 if len(mat2) == 0 else mat2

    if len(mat1[0]) != len(mat2[0]):
        return None

    return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
