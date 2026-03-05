#!/usr/bin/env python3
"""Q-learning implementation for FrozenLake environment."""

import numpy as np


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Perform Q-learning on a FrozenLake environment.

    Args:
        env: The ``FrozenLakeEnv`` instance from gymnasium.
        Q (numpy.ndarray): Initial Q-table of shape (n_states, n_actions).
        episodes (int): Number of episodes to train over.
        max_steps (int): Maximum steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Initial epsilon for epsilon-greedy policy.
        min_epsilon (float): Minimum value that epsilon can decay to.
        epsilon_decay (float): Decay amount for epsilon after each episode.

    Returns:
        tuple: ``(Q, total_rewards)`` where
            - Q is the updated Q-table.
            - total_rewards is a list containing the cumulative reward
              obtained in each episode.
    """

    # import locally to avoid circular dependencies
    epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # adjust reward if agent falls in a hole (terminal with no reward)
            if done and reward == 0:
                reward = -1

            # Q-learning update rule
            old_value = Q[state, action]
            next_max = np.max(Q[new_state])
            Q[state, action] = old_value + alpha * (
                reward + gamma * next_max - old_value)

            state = new_state
            episode_reward += reward

            if done:
                break

        # decay epsilon value
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        total_rewards.append(episode_reward)

    return Q, total_rewards


if __name__ == "__main__":
    # simple demonstration using helper functions in this folder
    load_env = __import__('0-load_env').load_frozen_lake
    env = load_env(map_name='4x4', is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    trained_Q, rewards = train(env, q_table, episodes=10, max_steps=50)
    print("Trained Q-table:\n", trained_Q)
    print("Rewards over episodes:", rewards)
