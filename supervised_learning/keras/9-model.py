#!/usr/bin/env python3
"""Module for saving and loading an entire model in Keras"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model."""
    network.save(filename)
    return None


def load_model(filename):
    """Loads an entire model."""
    model = K.models.load_model(filename)
    return model
