#!/usr/bin/env python3
"""Concatenate exchange data with source keys."""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """Return concatenated bitstamp/coinbase data with MultiIndex keys."""
    coinbase = index(df1)
    bitstamp = index(df2)

    bitstamp = bitstamp.loc[:1417411920]

    return pd.concat(
        [bitstamp, coinbase],
        keys=['bitstamp', 'coinbase']
    )
