#!/usr/bin/env python3
"""26. Persistence is Key"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network."""
        m = Y.shape[1]
        L = self.__L

        dZ = cache['A' + str(L)] - Y

        for layer in range(L, 0, -1):
            A_prev = cache['A' + str(layer - 1)]
            Wl = self.__weights['W' + str(layer)]
            b_l = self.__weights['b' + str(layer)]

            dW = (1 / m) * (dZ @ A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            Wl_before = Wl.copy()

            self.__weights['W' + str(layer)] = Wl - alpha * dW
            self.__weights['b' + str(layer)] = b_l - alpha * db

            if layer > 1:
                A_prev_act = cache['A' + str(layer - 1)]
                dZ = (Wl_before.T @ dZ) * (A_prev_act * (1 - A_prev_act))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iterations_list = []

        A, _ = self.forward_prop(X)
        initial_cost = self.cost(Y, A)

        if verbose:
            print(f"Cost after 0 iterations: {initial_cost}")

        if graph:
            costs.append(initial_cost)
            iterations_list.append(0)

        for i in range(1, iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if (verbose or graph) and (i % step == 0 or i == iterations):
                A, _ = self.forward_prop(X)
                current_cost = self.cost(Y, A)

                if verbose:
                    print(f"Cost after {i} iterations: {current_cost}")

                if graph:
                    costs.append(current_cost)
                    iterations_list.append(i)

        if graph:
            plt.figure(figsize=(10, 6))
            plt.plot(iterations_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the model to a file."""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a model from a file."""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
