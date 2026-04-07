#!/usr/bin/env python3
"""Load a DataFrame from a delimited file."""

import pandas as pd


def from_file(filename, delimiter):
    """Return a DataFrame loaded from filename using delimiter."""
    return pd.read_csv(filename, sep=delimiter)
