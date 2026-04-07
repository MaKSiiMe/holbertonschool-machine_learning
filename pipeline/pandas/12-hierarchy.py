#!/usr/bin/env python3
"""Create a chronologically ordered hierarchical DataFrame."""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """Concatenate exchanges on a timestamp-first MultiIndex."""
    coinbase = index(df1).loc[1417411980:1417417980]
    bitstamp = index(df2).loc[1417411980:1417417980]

    data = pd.concat(
        [bitstamp, coinbase],
        keys=['bitstamp', 'coinbase']
    )
    return data.swaplevel(0, 1).sort_index()
