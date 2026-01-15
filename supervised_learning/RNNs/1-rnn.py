#!/usr/bin/env python3
"""1. RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Effectue la propagation avant d'un RNN simple.
    rnn_cell: instance de RNNCell
    X: np.ndarray de forme (t, m, i)
    h_0: np.ndarray de forme (m, h)
    Retourne: H, Y
    H: np.ndarray de tous les états cachés (t+1, m, h)
    Y: np.ndarray de toutes les sorties (t, m, o)
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    # On suppose que la sortie y a la même dimension de sortie que rnn_cell.by
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    h_prev = h_0
    for step in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[step])
        H[step + 1] = h_next
        Y[step] = y
        h_prev = h_next
    return H, Y
