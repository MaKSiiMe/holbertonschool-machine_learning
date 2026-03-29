#!/usr/bin/env python3
"""
Module for computing policy gradients in reinforcement learning.
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


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient for a state and weight matrix.

    Parameters:
    state: np.ndarray - the current observation of the environment
    weight: np.ndarray - the weight matrix

    Returns:
    tuple: (action, gradient) where action is sampled from policy and
           gradient is the gradient with respect to the weights
    """
    # Ensure state is 2D (batch_size, features)
    if state.ndim == 1:
        state = state.reshape(1, -1)

    # Get policy probabilities using the policy function
    policy_probs = policy(state, weight)

    # Sample an action from the policy distribution
    action = np.random.choice(policy_probs.shape[1], p=policy_probs[0])

    # Compute the gradient of log probability
    # For softmax policy: grad = s^T @ (e_a - π)
    one_hot_action = np.zeros_like(policy_probs)
    one_hot_action[0, action] = 1

    # Gradient shape will match weight matrix shape
    gradient = state.T @ (one_hot_action - policy_probs)

    return action, gradient
