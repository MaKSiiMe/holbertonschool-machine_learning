#!/usr/bin/env python3
"""Slice specific DataFrame columns at fixed intervals."""


def slice(df):
    """Return High/Low/Close/Volume_(BTC) columns every 60th row."""
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
