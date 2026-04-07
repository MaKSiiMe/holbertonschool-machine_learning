#!/usr/bin/env python3
"""Sort DataFrame by High price."""


def high(df):
    """Return DataFrame sorted by High in descending order."""
    return df.sort_values(by="High", ascending=False)
