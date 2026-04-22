#!/usr/bin/env python3
"""Image brightness adjustment utility."""

import tensorflow as tf


def change_brightness(image, max_delta):
	"""Randomly change image brightness within +/- max_delta."""
	return tf.image.random_brightness(image, max_delta=max_delta)
