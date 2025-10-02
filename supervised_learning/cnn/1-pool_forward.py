#!/usr/bin/env python3
"""1. Pooling Forward Prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1

    # Initialize output
    A = np.zeros((m, output_h, output_w, c_prev))

    # Perform pooling
    for i in range(output_h):
        for j in range(output_w):
            start_i = i * sh
            start_j = j * sw

            # Extract the window
            window = A_prev[:, start_i:start_i+kh, start_j:start_j+kw, :]

            # Apply pooling operation
            if mode == 'max':
                A[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(window, axis=(1, 2))

    return A
