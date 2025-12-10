#!/usr/bin/env python3
"""Bayesian Optimization class"""

import numpy as np
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
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
