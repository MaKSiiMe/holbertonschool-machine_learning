#!/usr/bin/env python3
"""Normalization constants calculation module."""
import numpy as np


def normalization_constants(X):
    """Calculates the normalization (standardization) constants of a matrix."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    return mu, sigma
