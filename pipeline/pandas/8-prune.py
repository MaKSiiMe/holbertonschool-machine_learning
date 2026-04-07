#!/usr/bin/env python3
"""Remove rows with missing Close values."""


def prune(df):
    """Return DataFrame without rows where Close is NaN."""
    return df.dropna(subset=["Close"])
