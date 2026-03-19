#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm (first-visit)"""
    for episode in range(episodes):
        state = 0
        env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))
            if terminated or truncated:
                break
            state = new_state

        episode_data = np.array(episode_data, dtype=int)
        G = 0
        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            if state not in episode_data[:episode, 0]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
