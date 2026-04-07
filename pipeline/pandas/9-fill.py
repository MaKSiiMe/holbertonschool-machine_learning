#!/usr/bin/env python3
"""Fill missing values in a financial DataFrame."""


def fill(df):
    """Apply required cleaning and fill rules, then return DataFrame."""
    df = df.drop(columns=["Weighted_Price"])
    df["Close"] = df["Close"].ffill()

    for column in ["High", "Low", "Open"]:
        df[column] = df[column].fillna(df["Close"])

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
