#!/usr/bin/env python3
"""Module for calculating cofactor matrix"""


def determinant(matrix):
    """Calculate determinant of a matrix"""
    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        submatrix = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            submatrix.append(row)
        cofactor_val = ((-1) ** j) * matrix[0][j] * determinant(submatrix)
        det += cofactor_val

    return det


def minor(matrix):
    """Calculate minor matrix of a matrix"""
    n = len(matrix)

    if n == 1:
        return [[1]]

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            submatrix = []
            for row_idx in range(n):
                if row_idx != i:
                    row = []
                    for col_idx in range(n):
                        if col_idx != j:
                            row.append(matrix[row_idx][col_idx])
                    submatrix.append(row)
            minor_value = determinant(submatrix)
            minor_row.append(minor_value)
        minor_matrix.append(minor_row)

    return minor_matrix


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix

    Args:
        matrix: list of lists whose cofactor matrix should be calculated

    Returns:
        The cofactor matrix of matrix

    Raises:
        TypeError: if matrix is not a list of lists
        ValueError: if matrix is not a non-empty square matrix
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty
    if len(matrix) == 0 or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Get matrix dimensions
    n = len(matrix)

    # Check if matrix is square
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix, cofactor is [[1]]
    if n == 1:
        return [[1]]

    # Get minor matrix
    minor_matrix = minor(matrix)

    # Apply alternating signs to get cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            # Apply sign (-1)^(i+j)
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * minor_matrix[i][j])
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
