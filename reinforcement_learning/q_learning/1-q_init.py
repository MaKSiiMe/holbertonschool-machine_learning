#!/usr/bin/env python3
"""Q-table initialization for FrozenLake environment."""

import numpy as np


def q_init(env):
    """Initialize the Q-table for a given FrozenLake environment.

    Args:
        env: The ``FrozenLakeEnv`` instance from gymnasium.

    Returns:
        numpy.ndarray: A table of shape (nS, nA) filled with zeros where
        ``nS`` is the number of states and ``nA`` is the number of actions
        available in the environment.
    """

    # number of states from the observation space (assumed discrete)
    n_states = env.observation_space.n
    # number of actions from the action space (assumed discrete)
    n_actions = env.action_space.n

    # initialize the Q-table with zeros
    q_table = np.zeros((n_states, n_actions))

    return q_table
