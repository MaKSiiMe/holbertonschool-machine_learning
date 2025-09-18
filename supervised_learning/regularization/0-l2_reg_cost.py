#!/usr/bin/env python3
"""0. L2 Regularization Cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization"""
    weights_squared_sum = 0

    for layer in range(1, L + 1):
        W = weights[f'W{layer}']
        weights_squared_sum += np.sum(W ** 2)

    l2_term = (lambtha / (2 * m)) * weights_squared_sum

    return cost + l2_term
