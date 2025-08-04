#!/usr/bin/env python3
"""Module to concatenate two arrays."""


def cat_arrays(arr1, arr2):
    """Concatenate two arrays."""
    if not isinstance(arr1, list) or not isinstance(arr2, list):
        return None
    return arr1 + arr2
