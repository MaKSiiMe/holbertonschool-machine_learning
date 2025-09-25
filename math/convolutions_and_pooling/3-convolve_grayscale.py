#!/usr/bin/env python3
"""3. Strided Convolution"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    if ph > 0 or pw > 0:
        padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                               mode='constant', constant_values=0)
    else:
        padded_images = images

    padded_h, padded_w = padded_images.shape[1], padded_images.shape[2]
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1

    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            start_i = i * sh
            start_j = j * sw
            convolved[:, i, j] = np.sum(
                padded_images[:, start_i:start_i+kh,
                              start_j:start_j+kw] * kernel,
                axis=(1, 2)
            )

    return convolved
