#!/usr/bin/env python3
"""Module for plot the exponential decay of two radioactive elements."""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """Function to plot the exponential decay of two radioactive elements."""
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.plot(x, y1, color='red', label='C-14', linestyle='--')
    plt.plot(x, y2, color='green', label='Ra-226')
    plt.legend()
    plt.savefig("two_is_better.png")
    plt.show()
