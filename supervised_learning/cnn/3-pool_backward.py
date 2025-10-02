#!/usr/bin/env python3
"""3. Pooling Back Prop"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer"""
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize gradient
    dA_prev = np.zeros_like(A_prev)

    # Perform backpropagation
    for i in range(h_new):
        for j in range(w_new):
            start_i = i * sh
            start_j = j * sw

            if mode == 'max':
                # For max pooling, distribute gradient to max value
                for k in range(c):
                    for example in range(m):
                        # Extract the window
                        window = A_prev[example,
                                        start_i:start_i+kh,
                                        start_j:start_j+kw,
                                        k]

                        # Create mask where max value is
                        mask = (window == np.max(window))

                        # Distribute gradient to max position(s)
                        dA_prev[example,
                                start_i:start_i+kh,
                                start_j:start_j+kw,
                                k] += mask * dA[example, i, j, k]

            elif mode == 'avg':
                # For average pooling, distribute gradient uniformly
                avg_dA = dA[:, i, j, :] / (kh * kw)
                dA_prev[:,
                        start_i:start_i+kh,
                        start_j:start_j+kw,
                        :] += avg_dA[:, np.newaxis, np.newaxis, :]

    return dA_prev
