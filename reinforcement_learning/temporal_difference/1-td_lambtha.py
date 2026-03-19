#!/usr/bin/env python3
"""TD(lambda) algorithm for value estimation"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Performs the TD(lambda) algorithm with eligibility traces"""
    for episode in range(episodes):
        state, _ = env.reset()
        e = np.zeros(V.shape)

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[new_state] - V[state]
            e = gamma * lambtha * e
            e[state] += 1
            V = V + alpha * delta * e

            if terminated or truncated:
                break
            state = new_state

    return V
