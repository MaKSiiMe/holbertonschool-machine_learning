#!/usr/bin/env python3
"""Extract selected DataFrame rows as NumPy array."""


def array(df):
    """Return the last 10 rows of High and Close as a NumPy array."""
    return df[["High", "Close"]].tail(10).to_numpy()
