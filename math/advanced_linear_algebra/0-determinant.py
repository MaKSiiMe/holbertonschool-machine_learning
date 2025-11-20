#!/usr/bin/env python3
"""Module for calculating matrix determinant"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    Args:
        matrix: list of lists whose determinant should be calculated

    Returns:
        The determinant of matrix

    Raises:
        TypeError: if matrix is not a list of lists
        ValueError: if matrix is not square
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Handle empty matrix case [[]]
    if matrix == [[]]:
        return 1

    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Get matrix dimensions
    n = len(matrix)

    # Check if matrix is square
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case: nxn matrix (n > 2)
    # Use cofactor expansion along the first row
    det = 0
    for j in range(n):
        # Create submatrix by removing first row and j-th column
        submatrix = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            submatrix.append(row)

        # Calculate cofactor and add to determinant
        cofactor = ((-1) ** j) * matrix[0][j] * determinant(submatrix)
        det += cofactor

    return det
