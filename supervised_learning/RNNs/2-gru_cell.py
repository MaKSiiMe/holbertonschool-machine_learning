#!/usr/bin/env python3
"""2. GRU Cell"""
import numpy as np


class GRUCell:
    """Représente une cellule GRU (Gated Recurrent Unit)."""
    def __init__(self, i, h, o):
        """
        i: dimension de l'entrée
        h: dimension de l'état caché
        o: dimension de la sortie
        """
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Effectue la propagation avant pour un pas de temps.
        h_prev: (m, h) état caché précédent
        x_t: (m, i) entrée à l'instant t
        Retourne: h_next, y
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        # Update gate
        z = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))
        # Reset gate
        r = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))
        # Intermediate hidden state
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)
        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_hat
        # Output
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)
        return h_next, y
