#!/usr/bin/env python3
"""TD(lambda) algorithm for value estimation"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.9):
    """Performs the TD(lambda) algorithm with eligibility traces"""
    for _ in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibility *= gamma * lambtha
            eligibility[state] += 1
            V += alpha * delta * eligibility

            if terminated or truncated:
                break
            state = next_state

    return V
