#!/usr/bin/env python3
"""16. DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """Class that defines a deep neural network performing binary
    classification."""
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for layer in range(self.L):
            layer_size = layers[layer]
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            if layer == 0:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[layer - 1]

            self.weights['W' + str(layer + 1)] = (
                np.random.randn(layer_size, prev_layer_size) *
                np.sqrt(2 / prev_layer_size))
            self.weights['b' + str(layer + 1)] = np.zeros((layer_size, 1))
