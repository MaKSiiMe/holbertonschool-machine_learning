#!/usr/bin/env python3
"""Batch normalization module."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network
    using batch normalization."""
    mu = np.mean(Z, axis=0, keepdims=True)
    sigma_squared = np.var(Z, axis=0, keepdims=True)

    Z_norm = (Z - mu) / np.sqrt(sigma_squared + epsilon)

    Z_out = gamma * Z_norm + beta

    return Z_out
