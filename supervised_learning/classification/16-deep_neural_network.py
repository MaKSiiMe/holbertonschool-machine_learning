#!/usr/bin/env python3
"""16. DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """Class that defines a deep neural network performing binary classification."""
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0 or not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            layer_size = layers[l]
            if l == 0:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[l - 1]

            self.weights['W' + str(l + 1)] = np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
            self.weights['b' + str(l + 1)] = np.zeros((layer_size, 1))
