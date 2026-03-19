#!/usr/bin/env python3
"""Train a DQN agent on Atari Breakout and save weights to policy.h5."""

import time
import numpy as np
import gymnasium as gym
from PIL import Image
from gymnasium import spaces
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, Permute
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
import rl_patch  # noqa: F401
from rl.agents import DQNAgent
from rl.callbacks import Callback
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy


class PreprocessWrapper(gym.Wrapper):
    """Grayscale + resize + frame skip + max-pooling + episodic life."""

    def __init__(self, env, frame_skip=4, noop_max=30, episodic_life=True):
        """Initialize wrapper with frame skip count and noop max.

        Args:
            env: gymnasium environment to wrap.
            frame_skip: number of frames to repeat each action.
            noop_max: max random no-ops at episode start for stochasticity.
            episodic_life: if True, treat each life loss as episode end.
        """
        super().__init__(env)
        self._frame_skip = frame_skip
        self._noop_max = noop_max
        self._episodic_life = episodic_life
        self._lives = 0
        self._was_real_done = True
        self._prev_obs = None
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def _process(self, obs):
        """Convert RGB frame to 84x84 grayscale uint8 array."""
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        img = Image.fromarray(gray).resize((84, 84), Image.NEAREST)
        return np.array(img, dtype=np.uint8)

    def reset(self, **kwargs):
        """Reset env if game over; else noop-step to resume after life loss."""
        if self._was_real_done:
            obs, _ = self.env.reset(**kwargs)
            n_noops = self.np_random.integers(0, self._noop_max + 1)
            for _ in range(n_noops):
                obs, _, terminated, truncated, _ = self.env.step(0)
                if terminated or truncated:
                    obs, _ = self.env.reset(**kwargs)
        else:
            obs, _, _, _, _ = self.env.step(0)
        self._prev_obs = obs
        self._lives = self.env.unwrapped.ale.lives()
        return self._process(obs)

    def step(self, action):
        """Repeat action frame_skip steps; max-pool last 2 frames.

        Args:
            action: action index to repeat.

        Returns:
            Tuple of (obs, reward, done, info) with life-loss terminal signal.
        """
        total_reward = 0.0
        for _ in range(self._frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        obs_maxed = np.maximum(obs, self._prev_obs)
        self._prev_obs = obs
        self._was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        done = self._was_real_done or (
            self._episodic_life and lives < self._lives
        )
        self._lives = lives
        total_reward = np.clip(total_reward, -1.0, 1.0)
        return self._process(obs_maxed), total_reward, done, info


class CompatWrapper(gym.Wrapper):
    """Gymnasium wrapper that exposes the legacy Gym API for keras-rl2."""

    def reset(self, **kwargs):
        """Reset the environment and return only the observation."""
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            return obs[0]
        return obs

    def step(self, action):
        """Forward step; PreprocessWrapper already returns 4-tuple."""
        return self.env.step(action)

    def render(self, **kwargs):
        """Render, ignoring the legacy mode keyword from keras-rl2."""
        kwargs.pop("mode", None)
        return self.env.render(**kwargs)


class EpisodeLogCallback(Callback):
    """Print a training summary every interval episodes."""

    def __init__(self, interval=25, nb_steps_total=None):
        """Initialize with logging interval and optional total steps.

        Args:
            interval: number of episodes between each printed summary.
            nb_steps_total: total training steps, used to compute ETA.
        """
        self.interval = interval
        self._nb_steps_total = nb_steps_total
        self._rewards = []
        self._losses = []
        self._mean_qs = []
        self._start_time = None
        self._last_log_time = None
        self._last_log_step = 0

    def on_train_begin(self, logs):
        """Record training start time."""
        self._start_time = time.time()
        self._last_log_time = time.time()
        self._last_log_step = 0

    def on_step_end(self, step, logs):
        """Accumulate loss and mean_q from each training step."""
        metrics = logs.get("metrics", [])
        names = self.model.metrics_names
        if metrics is not None and len(metrics) == len(names):
            metric_map = dict(zip(names, metrics))
            loss = metric_map.get("loss")
            mq = metric_map.get("mean_q")
            if loss is not None and not np.isnan(loss):
                self._losses.append(loss)
            if mq is not None and not np.isnan(mq):
                self._mean_qs.append(mq)

    def on_episode_end(self, episode, logs):
        """Print summary every interval episodes."""
        self._rewards.append(logs.get("episode_reward", 0))
        if (episode + 1) % self.interval == 0:
            nb_steps = logs.get("nb_steps", 0)
            now = time.time()
            elapsed = now - self._start_time
            interval_time = now - self._last_log_time
            interval_steps = nb_steps - self._last_log_step
            sps = interval_steps / interval_time if interval_time > 0 else 0
            self._last_log_time = now
            self._last_log_step = nb_steps
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            if self._nb_steps_total and sps > 0:
                remaining = (self._nb_steps_total - nb_steps) / sps
                eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
            else:
                eta_str = "?"
            mean_short = np.mean(self._rewards[-self.interval:])
            mean_100 = np.mean(self._rewards[-100:])
            best = max(self._rewards)
            policy = self.model.policy
            if hasattr(policy, "value_max") and hasattr(policy, "nb_steps"):
                nb_pol = max(1, policy.nb_steps)
                eps = policy.value_max + (
                    policy.value_min - policy.value_max
                ) * min(1.0, float(self.model.step) / float(nb_pol))
            else:
                eps = float("nan")
            loss_str = (
                f"{np.mean(self._losses[-500:]):.4f}"
                if self._losses else "--"
            )
            mq_str = (
                f"{np.mean(self._mean_qs[-500:]):.3f}"
                if self._mean_qs else "--"
            )
            print(
                f"ep {episode + 1:6d} | "
                f"step {nb_steps:8d} | "
                f"t {elapsed_str} | "
                f"sps {sps:4.0f} | "
                f"eta {eta_str} | "
                f"eps {eps:.3f} | "
                f"loss {loss_str} | "
                f"mean_q {mq_str} | "
                f"mean-{self.interval}: {mean_short:5.2f} | "
                f"mean-100: {mean_100:5.2f} | "
                f"best: {best:.0f}"
            )


class SaveBestCallback(Callback):
    """Save DQN weights whenever the rolling mean episode reward improves."""

    def __init__(self, filepath="policy.h5", window=100):
        """Initialize with target filepath and averaging window size.

        Args:
            filepath: path where improved weights are saved.
            window: number of recent episodes used to compute the mean.
        """
        self.filepath = filepath
        self.window = window
        self._best = -np.inf
        self._rewards = []

    def on_episode_end(self, episode, logs):
        """Save weights if the rolling mean reward is a new best."""
        reward = logs.get("episode_reward", 0)
        self._rewards.append(reward)
        mean = np.mean(self._rewards[-self.window:])
        if mean > self._best:
            self._best = mean
            self.model.save_weights(self.filepath, overwrite=True)
            print(
                f"\nBest model saved at episode {episode} "
                f"(mean-{self.window}: {mean:.2f})"
            )


class EarlyStoppingCallback(Callback):
    """Stop training when rolling mean episode reward exceeds threshold."""

    def __init__(self, threshold=30.0, patience=20):
        """Initialize with reward threshold and patience window.

        Args:
            threshold: mean reward value that triggers early stop.
            patience: number of recent episodes used to compute the mean.
        """
        self.threshold = threshold
        self.patience = patience
        self._rewards = []
        self._triggered = False

    def on_episode_end(self, episode, logs):
        """Raise KeyboardInterrupt to stop fit() if threshold is reached."""
        reward = logs.get("episode_reward", 0)
        self._rewards.append(reward)
        if len(self._rewards) >= self.patience:
            mean = np.mean(self._rewards[-self.patience:])
            if mean >= self.threshold:
                self._triggered = True
                raise KeyboardInterrupt

    def on_train_end(self, logs):
        """Print message if training was stopped by this callback."""
        if self._triggered:
            mean = np.mean(self._rewards[-self.patience:])
            print(
                f"\nEarly stop: mean reward "
                f"{mean:.1f} >= {self.threshold}"
            )


def build_model(input_shape, nb_actions, window_length=4):
    """Build the Nature DQN convolutional model.

    Stacks window_length frames and uses Permute to convert
    (window, H, W) -> (H, W, window) before the Conv2D layers.

    Args:
        input_shape: shape of a single observation frame (e.g. (84, 84)).
        nb_actions: number of discrete actions available.
        window_length: number of stacked frames fed to the network.

    Returns:
        A compiled-ready Sequential Keras model.
    """
    model = Sequential()
    model.add(Permute(
        (2, 3, 1),
        input_shape=(window_length,) + input_shape
    ))
    model.add(Lambda(lambda x: x / 255.0))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(nb_actions, activation="linear"))
    return model


def make_env(render_mode="rgb_array", episodic_life=True):
    """Create and wrap the Breakout environment.

    Applies grayscale + 84x84 resize + frame skip + max-pooling + episodic
    life (via PreprocessWrapper), then CompatWrapper for keras-rl2.

    Args:
        render_mode: render mode passed to gymnasium.make.
        episodic_life: treat life loss as episode end (True for training).

    Returns:
        Wrapped Breakout environment ready for keras-rl2.
    """
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
    env = PreprocessWrapper(env, frame_skip=4, episodic_life=episodic_life)
    env = CompatWrapper(env)
    return env


def main():
    """Create Breakout environment, train DQN agent, and save weights."""
    env = make_env(render_mode="rgb_array")

    nb_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    window_length = 4
    nb_steps = 100000

    model = build_model(obs_shape, nb_actions, window_length)

    memory = SequentialMemory(limit=200000, window_length=window_length)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.01,
        nb_steps=nb_steps // 4,
    )

    lr_schedule = CosineDecay(
        initial_learning_rate=0.00025,
        decay_steps=nb_steps // 4,
        alpha=0.05,
    )

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
    dqn.compile(Adam(learning_rate=lr_schedule), metrics=["mae"])

    callbacks = [
        EpisodeLogCallback(interval=25, nb_steps_total=nb_steps),
        SaveBestCallback(filepath="policy.h5"),
        EarlyStoppingCallback(threshold=50.0, patience=50),
    ]
    dqn.fit(
        env, nb_steps=nb_steps, visualize=False, verbose=0,
        callbacks=callbacks
    )

    dqn.save_weights("policy.h5", overwrite=True)
    env.close()


if __name__ == "__main__":
    main()
