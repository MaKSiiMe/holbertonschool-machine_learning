#!/usr/bin/env python3
"""
This module contains functions for computing policy gradients in reinforcement learning.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight matrix using softmax.

    Parameters:
    matrix: np.ndarray of shape (1, n) - the state
    weight: np.ndarray of shape (n, m) - the weight matrix

    Returns:
    np.ndarray of shape (1, m) - the policy (action probabilities)
    """
    # Compute logits: state @ weight
    logits = np.dot(matrix, weight)

    # Apply softmax to get probabilities
    e_logits = np.exp(logits)
    policy_probs = e_logits / np.sum(e_logits, axis=1, keepdims=True)

    return policy_probs
