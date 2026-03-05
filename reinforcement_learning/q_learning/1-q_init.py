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

    # number of states from the observation space (assumed discrete).
    # ``FrozenLakeEnv`` correctly sets ``observation_space.n`` based on
    # the underlying map, but some external code (or future gym versions)
    # may occasionally misreport it.  In those cases we can fall back to
    # the size of the transition dictionary ``P`` stored on the wrapped
    # environment, which always has one entry per state.
    try:
        n_states = env.observation_space.n
    except AttributeError:
        # last resort: count entries in the transition matrix
        n_states = len(env.unwrapped.P)

    # number of actions from the action space (assumed discrete)
    n_actions = env.action_space.n

    # initialize the Q-table with zeros
    q_table = np.zeros((n_states, n_actions))

    return q_table
