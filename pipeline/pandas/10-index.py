#!/usr/bin/env python3
"""Set Timestamp as DataFrame index."""


def index(df):
    """Return DataFrame with Timestamp set as index."""
    return df.set_index("Timestamp")
