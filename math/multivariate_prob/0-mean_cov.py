#!/usr/bin/env python3
"""Module for calculating mean and covariance"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
           n is the number of data points
           d is the number of dimensions in each data point

    Returns:
        mean: numpy.ndarray of shape (1, d) containing the mean
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix

    Raises:
        TypeError: if X is not a 2D numpy.ndarray
        ValueError: if X contains less than 2 data points
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")

    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    cov = np.matmul(X_centered.T, X_centered) / n

    return mean, cov
