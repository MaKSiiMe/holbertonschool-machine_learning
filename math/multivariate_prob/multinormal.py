#!/usr/bin/env python3
"""Module for Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """
        Initialize Multivariate Normal distribution

        Args:
            data: numpy.ndarray of shape (d, n) containing the data set
                  n is the number of data points
                  d is the number of dimensions in each data point

        Raises:
            TypeError: if data is not a 2D numpy.ndarray
            ValueError: if data contains less than 2 data points
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")

        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point

        Returns:
            The value of the PDF

        Raises:
            TypeError: if x is not a numpy.ndarray
            ValueError: if x does not have the correct shape
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        diff = x - self.mean
        exponent = -0.5 * np.matmul(np.matmul(diff.T, inv_cov), diff)
        pi = np.pi
        coefficient = 1 / np.sqrt(((2 * pi) ** d) * det_cov)
        pdf_value = coefficient * np.exp(exponent)

        return pdf_value[0][0]
