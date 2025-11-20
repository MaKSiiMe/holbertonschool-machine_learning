#!/usr/bin/env python3
"""Module for calculating matrix definiteness"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix

    Args:
        matrix: numpy.ndarray of shape (n, n) whose definiteness should be
                calculated

    Returns:
        String indicating definiteness or None

    Raises:
        TypeError: if matrix is not a numpy.ndarray
    """
    # Check if matrix is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is 2D
    if len(matrix.shape) != 2:
        return None

    # Check if matrix is square
    n, m = matrix.shape
    if n != m or n == 0:
        return None

    # Check if matrix is symmetric (necessary for definiteness)
    if not np.allclose(matrix, matrix.T):
        return None

    # Calculate eigenvalues
    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except Exception:
        return None

    # Use a small tolerance for numerical comparison
    tol = 1e-10

    # Count positive, negative, and zero eigenvalues
    positive = np.sum(eigenvalues > tol)
    negative = np.sum(eigenvalues < -tol)
    zero = np.sum(np.abs(eigenvalues) <= tol)

    # Determine definiteness
    if positive == n:
        return "Positive definite"
    elif negative == n:
        return "Negative definite"
    elif positive + zero == n and zero > 0:
        return "Positive semi-definite"
    elif negative + zero == n and zero > 0:
        return "Negative semi-definite"
    elif positive > 0 and negative > 0:
        return "Indefinite"
    else:
        return None
