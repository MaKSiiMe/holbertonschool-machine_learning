#!/usr/bin/env python3
"""6. Pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    pooled = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            start_i = i * sh
            start_j = j * sw

            pool_region = images[:, start_i:start_i+kh, start_j:start_j+kw, :]

            if mode == 'max':
                pooled[:, i, j, :] = np.max(pool_region, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(pool_region, axis=(1, 2))

    return pooled
