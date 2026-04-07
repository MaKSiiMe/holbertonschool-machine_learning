#!/usr/bin/env python3
"""Create a pandas DataFrame from a NumPy array."""

import pandas as pd


def from_numpy(array):
    """Return a DataFrame with alphabetical uppercase column labels."""
    cols = [chr(ord('A') + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=cols)
