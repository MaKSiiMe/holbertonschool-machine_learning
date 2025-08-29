#!/usr/bin/env python3
"""19. DeepNeuralNetwork Cost"""
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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer in range(self.__L):
            layer_size = layers[layer]
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            if layer == 0:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[layer - 1]

            self.__weights['W' + str(layer + 1)] = (
                np.random.randn(layer_size, prev_layer_size) *
                np.sqrt(2 / prev_layer_size))
            self.__weights['b' + str(layer + 1)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.__cache['A0'] = X

        for layer in range(1, self.__L + 1):
            A_prev = self.__cache['A' + str(layer - 1)]
            W = self.__weights['W' + str(layer)]
            b = self.__weights['b' + str(layer)]

            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(layer)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression."""
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return float(cost)
