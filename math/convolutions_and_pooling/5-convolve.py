#!/usr/bin/env python3
"""5. Multiple Kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels"""
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h + 1) // 2
        pw = ((w - 1) * sw + kw - w + 1) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    if ph > 0 or pw > 0:
        padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                               mode='constant', constant_values=0)
    else:
        padded_images = images

    padded_h, padded_w = padded_images.shape[1], padded_images.shape[2]
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1

    convolved = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                start_i = i * sh
                start_j = j * sw
                convolved[:, i, j, k] = np.sum(
                    padded_images[:, start_i:start_i+kh,
                                  start_j:start_j+kw, :] *
                    kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return convolved
