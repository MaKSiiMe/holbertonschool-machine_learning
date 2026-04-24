#!/usr/bin/env python3
"""Flip image horizontally."""

import tensorflow as tf


def flip_image(image):
    """Flip an image horizontally."""
    return tf.image.flip_left_right(image)
