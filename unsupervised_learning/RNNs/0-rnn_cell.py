#!/usr/bin/env python3
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """
        i: dimension of input
        h: dimension of hidden state
        o: dimension of output
        """
        # Poids pour l'état caché (concaténé [h_prev, x_t])
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        # Poids pour la sortie
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        h_prev: (m, h) état caché précédent
        x_t: (m, i) entrée à l'instant t
        Retourne: h_next, y
        """
        # Concaténation de h_prev et x_t
        concat = np.concatenate((h_prev, x_t), axis=1)
        # Calcul du nouvel état caché
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        # Calcul de la sortie
        y_linear = np.matmul(h_next, self.Wy) + self.by
        # Softmax sur la sortie
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)
        return h_next, y
