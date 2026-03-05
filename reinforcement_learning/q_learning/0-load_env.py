#!/usr/bin/env python3
"""Load a FrozenLake environment from gymnasium."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the FrozenLake environment.

    Args:
        desc (list[list[str]] | None): Custom map description. If provided,
            this takes precedence over ``map_name``.
        map_name (str | None): Name of a built-in map (e.g. ``"4x4"`` or
            ``"8x8"``). Ignored if ``desc`` is given. When both ``desc`` and
            ``map_name`` are ``None`` the environment will generate a random
            8x8 map as per gymnasium's default behaviour.
        is_slippery (bool): Whether the frozen lake is slippery (stochastic
            transitions) or deterministic.

    Returns:
        gymnasium.Env: An instance of ``FrozenLake-v1`` configured with the
        requested parameters.
    """

    kwargs = {"is_slippery": is_slippery, "render_mode": "ansi"}
    if desc is not None:
        kwargs["desc"] = desc
    elif map_name is not None:
        kwargs["map_name"] = map_name

    return gym.make("FrozenLake-v1", **kwargs)
