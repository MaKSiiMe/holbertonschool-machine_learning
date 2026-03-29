#!/usr/bin/env python3
"""
Module implementing training for policy gradient agents.
"""

import numpy as np


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Trains a policy gradient agent on the given environment.

    Parameters:
    env: the environment to train on
    nb_episodes: number of episodes to train for
    alpha: learning rate
    gamma: discount factor

    Returns:
    list of scores (sum of rewards per episode)
    """
    policy_gradient = __import__('policy_gradient').policy_gradient

    scores = []

    # Initialize weights randomly
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    weight = np.random.rand(state_size, action_size)

    for episode in range(nb_episodes):
        state, _ = env.reset()
        score = 0

        # Collect trajectories
        trajectory = []

        # Run the episode
        done = False
        while not done:
            action, grad = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            trajectory.append((grad, reward))
            score += reward
            state = next_state

        # Calculate returns and update weights
        G = 0
        for t in range(len(trajectory) - 1, -1, -1):
            grad, reward = trajectory[t]
            G = reward + gamma * G
            weight += alpha * G * grad

        # Print episode info
        print("Episode: {} Score: {}".format(episode, score))
        scores.append(score)

    return scores
