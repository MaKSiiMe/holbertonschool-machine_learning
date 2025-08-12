#!/usr/bin/env python3
"""Module for plotting a line graph."""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Function to plot a line graph."""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.xlim(0, 10)
    plt.plot(y, color='red')
    plt.savefig('line_graph.png')
    plt.show()
