#!/usr/bin/env python3
"""Load a trained Breakout DQN policy and display played episodes."""

from tensorflow.keras.optimizers.legacy import Adam
import rl_patch  # noqa: F401
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from train import build_model, make_env


def main():
    """Load policy.h5 and run 5 full test episodes of Atari Breakout."""
    env = make_env(render_mode="human", episodic_life=False)

    nb_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    window_length = 4

    model = build_model(obs_shape, nb_actions, window_length)
    memory = SequentialMemory(limit=200000, window_length=window_length)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=10000,
        target_model_update=0.005,
        policy=policy,
        batch_size=128,
        train_interval=4,
        delta_clip=1.0,
        gamma=0.99,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    dqn.load_weights("policy.h5")
    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()


if __name__ == "__main__":
    main()
