#!/usr/bin/env python3
"""Gaussian Process class"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes a Gaussian Process.

        Args:
            X_init: numpy.ndarray of shape (t, 1) representing inputs
                    already sampled with the black-box function
            Y_init: numpy.ndarray of shape (t, 1) representing outputs
                    of the black-box function
            l: length parameter with the kernel
            sigma_f: standard deviation given to the output of the
                     black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices.

        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            Covariance kernel matrix as numpy.ndarray of shape (m, n)
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a GP.

        Args:
            X_s: numpy.ndarray of shape (s, 1) containing points whose
                 mean and standard deviation should be calculated

        Returns:
            mu: numpy.ndarray of shape (s,) containing the mean
            sigma: numpy.ndarray of shape (s,) containing the variance
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.Y
        mu = mu.reshape(-1)

        cov = K_ss - K_s.T @ K_inv @ K_s
        sigma = np.diagonal(cov)

        return mu, sigma
