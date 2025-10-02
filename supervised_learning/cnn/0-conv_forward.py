#!/usr/bin/env python3
"""0. Convolutional Forward Prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = max((h_prev - 1) * sh + kh - h_prev, 0) // 2
        pw = max((w_prev - 1) * sw + kw - w_prev, 0) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    if ph > 0 or pw > 0:
        A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                         mode='constant', constant_values=0)
    else:
        A_padded = A_prev

    padded_h, padded_w = A_padded.shape[1], A_padded.shape[2]
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1

    Z = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(c_new):
                start_i = i * sh
                start_j = j * sw
                Z[:, i, j, k] = np.sum(
                    A_padded[:, start_i:start_i+kh,
                            start_j:start_j+kw, :] *
                    W[:, :, :, k],
                    axis=(1, 2, 3)
                )

    Z = Z + b
    A = activation(Z)

    return A
