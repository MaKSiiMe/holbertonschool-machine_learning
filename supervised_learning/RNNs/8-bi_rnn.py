#!/usr/bin/env python3
"""8. Bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Effectue la propagation avant d'un RNN bidirectionnel.
    bi_cell: instance de BidirectionalCell
    X: (t, m, i) données d'entrée
    h_0: (m, h) état caché initial (avant)
    h_t: (m, h) état caché initial (arrière)
    Retourne: H, Y
    H: (t, m, 2*h) états cachés concaténés
    Y: (t, m, o) sorties
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    # Forward direction
    Hf = np.zeros((t, m, h))
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        Hf[step] = h_prev

    # Backward direction
    Hb = np.zeros((t, m, h))
    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        Hb[step] = h_next

    # Concaténer les états cachés
    H = np.concatenate((Hf, Hb), axis=2)

    # Calculer les sorties
    Y = bi_cell.output(H)
    return H, Y
