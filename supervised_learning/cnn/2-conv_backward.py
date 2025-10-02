#!/usr/bin/env python3
"""2. Convolutional Back Prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        ph = max((h_prev - 1) * sh + kh - h_prev, 0) // 2
        pw = max((w_prev - 1) * sw + kw - w_prev, 0) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    # Apply padding to A_prev
    if ph > 0 or pw > 0:
        A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          mode='constant', constant_values=0)
    else:
        A_padded = A_prev

    # Initialize gradients
    dA_padded = np.zeros_like(A_padded)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Perform backpropagation
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                start_i = i * sh
                start_j = j * sw

                # Gradient with respect to W
                dW[:, :, :, k] += np.sum(
                    A_padded[:, start_i:start_i+kh, start_j:start_j+kw, :] *
                    dZ[:, i:i+1, j:j+1, k:k+1],
                    axis=0
                )

                # Gradient with respect to A_prev
                for example in range(m):
                    dA_padded[example, start_i:start_i+kh,
                              start_j:start_j+kw, :] += (
                        W[:, :, :, k] * dZ[example, i, j, k]
                    )

    # Remove padding from dA_padded if necessary
    if ph > 0 or pw > 0:
        dA_prev = dA_padded[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_padded

    return dA_prev, dW, db
