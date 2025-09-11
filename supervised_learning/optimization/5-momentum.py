#!/usr/bin/env python3
"""Gradient descent with momentum optimization algorithm."""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient descent
    with momentum optimization algorithm."""
    v_new = beta1 * v + (1 - beta1) * grad

    var_updated = var - alpha * v_new

    return var_updated, v_new
