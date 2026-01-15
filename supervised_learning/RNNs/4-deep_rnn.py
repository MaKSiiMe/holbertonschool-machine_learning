#!/usr/bin/env python3
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Effectue la propagation avant d'un deep RNN.
    rnn_cells: liste d'instances de RNNCell (longueur num_layers)
    X: np.ndarray de forme (t, m, i)
    h_0: np.ndarray de forme (num_layers, m, h)
    Retourne: H, Y
    H: np.ndarray de tous les états cachés (t+1, num_layers, m, h)
    Y: np.ndarray de toutes les sorties (t, m, o)
    """
    t, m, _ = X.shape
    num_layers = len(rnn_cells)
    h = h_0.shape[2]
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t + 1, num_layers, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        x = X[step]
        h_list = []
        for layer in range(num_layers):
            h_prev = H[step, layer]
            cell = rnn_cells[layer]
            h_next, y = cell.forward(h_prev, x)
            h_list.append(h_next)
            x = h_next
        for layer in range(num_layers):
            H[step + 1, layer] = h_list[layer]
        Y[step] = y  # la sortie finale est celle de la dernière couche
    return H, Y
