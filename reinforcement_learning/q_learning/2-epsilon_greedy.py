#!/usr/bin/env python3
"""Epsilon-greedy action selection for Q-learning."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Select the next action using an epsilon-greedy policy.

    Args:
        Q (numpy.ndarray): The Q-table with shape (n_states, n_actions).
        state (int): The current state index.
        epsilon (float): The probability of choosing a random action (explore).

    Returns:
        int: Index of the action selected.
    """

    # number of actions available in the environment
    n_actions = Q.shape[1]

    # sample a probability to decide exploration vs exploitation
    p = np.random.uniform(0, 1)
    if p < epsilon:
        # explore: pick a random action
        return np.random.randint(n_actions)
    # exploit: choose the action with highest Q value for the current state
    return int(np.argmax(Q[state]))


if __name__ == "__main__":
    # quick demo when run as a script
    q = np.zeros((16, 4))
    state = 0
    eps = 0.1
    print("Chosen action:", epsilon_greedy(q, state, eps))
