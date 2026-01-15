#!/usr/bin/env python3
"""6. Bidirectional Cell Backward"""
import numpy as np


class BidirectionalCell:
    """Représente une cellule bidirectionnelle d'un RNN."""
    def __init__(self, i, h, o):
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calcule l'état caché dans la direction arrière pour un pas de temps.
        h_next: (m, h) état caché suivant
        x_t: (m, i) entrée à l'instant t
        Retourne: h_prev, l'état caché précédent
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)
        return h_prev
