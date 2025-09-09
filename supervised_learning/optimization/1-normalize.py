#!/usr/bin/env python3
"""Normalization module."""


def normalize(X, m, s):
    """Normalizes the dataset X using the provided mean and std deviation."""
    return (X - m) / s
