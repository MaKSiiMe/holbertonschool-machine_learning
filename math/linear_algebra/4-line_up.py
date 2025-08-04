#!/usr/bin/env python3
"""Module to add two arrays element-wise."""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise."""
    if len(arr1) != len(arr2):
        return None
    return [x + y for x, y in zip(arr1, arr2)]
