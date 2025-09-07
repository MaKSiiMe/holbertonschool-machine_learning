#!/usr/bin/env python3
"""Module for saving and loading a model's weights in Keras"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Saves a model's weights."""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Loads a model's weights."""
    network.load_weights(filename)
    return None
