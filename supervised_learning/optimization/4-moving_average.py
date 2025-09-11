#!/usr/bin/env python3
"""Weighted moving average calculation module with bias correction."""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set
    using bias correction."""
    moving_averages = []
    v = 0

    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** t)
        moving_averages.append(v_corrected)

    return moving_averages
