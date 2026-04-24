#!/usr/bin/env python3
"""Image hue adjustment utility."""

import tensorflow as tf


def change_hue(image, delta):
    """Change image hue by the provided delta."""
    return tf.image.adjust_hue(image, delta)
