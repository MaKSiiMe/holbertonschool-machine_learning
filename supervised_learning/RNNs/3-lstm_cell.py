#!/usr/bin/env python3
"""3. LSTM Cell"""
import numpy as np


class LSTMCell:
    """Représente une cellule LSTM (Long Short-Term Memory)."""
    def __init__(self, i, h, o):
        """
        i: dimension de l'entrée
        h: dimension de l'état caché
        o: dimension de la sortie
        """
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Effectue la propagation avant pour un pas de temps.
        h_prev: (m, h) état caché précédent
        c_prev: (m, h) état de cellule précédent
        x_t: (m, i) entrée à l'instant t
        Retourne: h_next, c_next, y
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        # Oubli
        f = 1 / (1 + np.exp(-(np.matmul(concat, self.Wf) + self.bf)))
        # Update (input gate)
        u = 1 / (1 + np.exp(-(np.matmul(concat, self.Wu) + self.bu)))
        # Cell candidate
        c_hat = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        # Output gate
        o = 1 / (1 + np.exp(-(np.matmul(concat, self.Wo) + self.bo)))
        # Next cell state
        c_next = f * c_prev + u * c_hat
        # Next hidden state
        h_next = o * np.tanh(c_next)
        # Output
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)
        return h_next, c_next, y
