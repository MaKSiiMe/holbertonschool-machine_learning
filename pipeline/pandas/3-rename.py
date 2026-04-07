#!/usr/bin/env python3
"""Rename and format DataFrame columns."""

import pandas as pd


def rename(df):
    """Rename Timestamp to Datetime and keep Datetime and Close."""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df[["Datetime", "Close"]]
