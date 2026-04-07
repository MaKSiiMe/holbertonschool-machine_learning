#!/usr/bin/env python3
"""Compute descriptive statistics for a DataFrame."""


def analyze(df):
    """Return descriptive statistics for all columns except Timestamp."""
    return df.drop(columns=["Timestamp"]).describe()
