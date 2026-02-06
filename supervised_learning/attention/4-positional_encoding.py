#!/usr/bin/env python3
"""Encodage positionnel pour transformer"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Retourne un ndarray (max_seq_len, dm) contenant les encodages positionnels.
    """
    # positions : (max_seq_len, 1)
    pos = np.arange(max_seq_len)[:, np.newaxis]
    # dims : (1, dm)
    i = np.arange(dm)[np.newaxis, :]
    # taux d'angle : pos / (10000^(2*(i//2)/dm))
    angle_rates = pos / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    # appliquer sin aux indices pairs, cos aux impairs
    pe = np.zeros((max_seq_len, dm))
    pe[:, 0::2] = np.sin(angle_rates[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rates[:, 1::2])
    return pe
