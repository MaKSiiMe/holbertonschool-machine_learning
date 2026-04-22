#!/usr/bin/env python3
"""Random crop utility."""

import tensorflow as tf


def crop_image(image, size):
	"""Perform a random crop on an image tensor."""
	return tf.image.random_crop(image, size=size)
