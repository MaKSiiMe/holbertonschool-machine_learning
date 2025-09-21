#!/usr/bin/env python3
"""5. Gradient Descent with Dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network
    with Dropout regularization using gradient descent"""
    m = Y.shape[1]

    dZ = cache[f'A{L}'] - Y

    for layer in range(L, 0, -1):
        A_prev = cache[f'A{layer-1}']

        dW = (1/m) * np.matmul(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            dA_prev = np.matmul(weights[f'W{layer}'].T, dZ)

            dropout_mask = cache[f'D{layer-1}']
            dA_prev = dA_prev * dropout_mask

            dA_prev = dA_prev / keep_prob

            dZ = dA_prev * (1 - A_prev**2)

        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * db
