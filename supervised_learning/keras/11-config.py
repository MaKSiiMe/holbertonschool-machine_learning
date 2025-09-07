#!/usr/bin/env python3
"""Module for saving and loading a model's configuration in Keras"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration in JSON format."""
    config_json = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(config_json)
    return None


def load_config(filename):
    """Loads a model with a specific configuration."""
    with open(filename, 'r') as json_file:
        config_json = json_file.read()
    model = K.models.model_from_json(config_json)
    return model
