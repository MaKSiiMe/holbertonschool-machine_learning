#!/usr/bin/env python3
"""Play function for a trained Q-learning agent on FrozenLake."""

import numpy as np


def play(env, Q, max_steps=100):
    """Have the trained agent play an episode and render states.

    Args:
        env: The ``FrozenLakeEnv`` instance from gymnasium (must be created
            with ``render_mode="ansi"`` so that ``render()`` returns a
            string).
        Q (numpy.ndarray): The Q-table used for selecting actions (always
            exploiting).
        max_steps (int): Maximum number of steps to execute in the episode.

    Returns:
        tuple: ``(total_reward, renders)`` where ``total_reward`` is the total
            reward obtained (returned as a float) and ``renders`` is a list
            of strings representing the board state after each step,
            including the initial state.
    """

    state, _ = env.reset()
    total_reward = 0
    renders = []

    # capture the initial board state
    first = env.render()
    renders.append(first)

    for _ in range(max_steps):
        # always exploit the Q-table
        action = int(np.argmax(Q[state]))
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward

        out = env.render()
        renders.append(out)

        if done:
            break

    # ensure we return a native python float for consistency
    return float(total_reward), renders


if __name__ == "__main__":
    # simple demonstration using helper functions
    load_env = __import__("0-load_env").load_frozen_lake
    env = load_env(map_name="4x4", is_slippery=False)
    # initialize a dummy Q-table with zeros
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    reward, boards = play(env, Q)
    print("Episode reward:", reward)
    print("Rendered steps:", len(boards))
