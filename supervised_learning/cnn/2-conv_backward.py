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
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0

    # Apply padding to A_prev
    A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant', constant_values=0)

    # Initialize gradients
    dA_padded = np.pad(np.zeros_like(A_prev),
                       ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                       mode='constant', constant_values=0)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Perform backpropagation
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = A_padded[i, vert_start:vert_end,
                                       horiz_start:horiz_end, :]

                    dA_padded[i, vert_start:vert_end,
                              horiz_start:horiz_end, :] += (
                        W[:, :, :, k] * dZ[i, h, w, k]
                    )

                    dW[:, :, :, k] += a_slice * dZ[i, h, w, k]

    # Remove padding from dA_padded if necessary
    if ph > 0 and pw > 0:
        dA_prev = dA_padded[:, ph:-ph, pw:-pw, :]
    elif ph > 0:
        dA_prev = dA_padded[:, ph:-ph, :, :]
    elif pw > 0:
        dA_prev = dA_padded[:, :, pw:-pw, :]
    else:
        dA_prev = dA_padded

    return dA_prev, dW, db
