#!/usr/bin/env python3
"""4. Forward Propagation with Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X

    for layer in range(1, L + 1):
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']

        A_prev = cache[f'A{layer-1}']

        Z = np.matmul(W, A_prev) + b

        if layer == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            A = np.tanh(Z)

            dropout_mask = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob).astype(int)

            A = A * dropout_mask

            A = A / keep_prob

            cache[f'D{layer}'] = dropout_mask

        cache[f'A{layer}'] = A

    return cache
