#!/usr/bin/env python3
"""Concatenate exchange data with source keys."""

index = __import__('10-index').index


def concat(df1, df2):
    """Return concatenated bitstamp/coinbase data with MultiIndex keys."""
    coinbase = index(df1)
    bitstamp = index(df2)

    bitstamp = bitstamp.loc[:1417411920]

    return __import__('pandas').concat(
        [bitstamp, coinbase],
        keys=['bitstamp', 'coinbase']
    )
