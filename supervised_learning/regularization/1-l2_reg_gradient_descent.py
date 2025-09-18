#!/usr/bin/env python3
"""1. Gradient Descent with L2 Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network
    using gradient descent with L2 regularization"""
    m = Y.shape[1]

    dZ = cache[f'A{L}'] - Y

    for layer in range(L, 0, -1):
        A_prev = cache[f'A{layer-1}'] if layer > 1 else cache['A0']

        dW = (1/m) * np.matmul(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        dW += (lambtha/m) * weights[f'W{layer}']

        if layer > 1:
            dA_prev = np.matmul(weights[f'W{layer}'].T, dZ)
            dZ = dA_prev * (1 - A_prev**2)

        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * db
        dZ = dA_prev * (1 - A_prev**2)
