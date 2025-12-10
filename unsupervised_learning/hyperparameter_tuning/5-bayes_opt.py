#!/usr/bin/env python3
"""Bayesian Optimization class"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes Bayesian Optimization.

        Args:
            f: black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1) representing inputs
                    already sampled with the black-box function
            Y_init: numpy.ndarray of shape (t, 1) representing outputs
                    of the black-box function
            bounds: tuple of (min, max) representing the bounds of the space
            ac_samples: number of samples that should be analyzed during
                        acquisition
            l: length parameter with the kernel
            sigma_f: standard deviation given to the output of the
                     black-box function
            xsi: exploration-exploitation factor with acquisition
            minimize: bool determining whether optimization should be
                      performed with minimization (True) or maximization
                      (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement.

        Returns:
            X_next: numpy.ndarray of shape (1,) representing the next best
                    sample point
            EI: numpy.ndarray of shape (ac_samples,) containing the expected
                improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            Y_sample_opt = np.min(self.gp.Y)
            improvement = Y_sample_opt - mu - self.xsi
        else:
            Y_sample_opt = np.max(self.gp.Y)
            improvement = mu - Y_sample_opt - self.xsi

        with np.errstate(divide='warn'):
            Z = improvement / sigma
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function.

        Args:
            iterations: maximum number of iterations to perform

        Returns:
            X_opt: numpy.ndarray of shape (1,) representing the optimal point
            Y_opt: numpy.ndarray of shape (1,) representing the optimal
                   function value
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.isclose(self.gp.X, X_next)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
