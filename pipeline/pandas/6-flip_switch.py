#!/usr/bin/env python3
"""Reverse and transpose a DataFrame."""


def flip_switch(df):
    """Sort by index descending, then transpose the DataFrame."""
    return df.sort_index(ascending=False).transpose()
