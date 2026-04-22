#!/usr/bin/env python3
"""Image contrast adjustment utility."""

import tensorflow as tf


def change_contrast(image, lower, upper):
	"""Randomly adjust image contrast within [lower, upper]."""
	return tf.image.random_contrast(image, lower=lower, upper=upper)
