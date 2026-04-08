#!/usr/bin/env python3
"""Display the number of SpaceX launches per rocket."""

from collections import Counter

import requests


def main():
    """Fetch launches and print launch counts per rocket name."""
    launches = requests.get(
        "https://api.spacexdata.com/v4/launches", timeout=10
    ).json()
    rockets = requests.get(
        "https://api.spacexdata.com/v4/rockets", timeout=10
    ).json()

    rocket_names = {rocket.get("id"): rocket.get("name") for rocket in rockets}
    counts = Counter(launch.get("rocket") for launch in launches)

    items = []
    for rocket_id, count in counts.items():
        name = rocket_names.get(rocket_id)
        if name:
            items.append((name, count))

    for name, count in sorted(items, key=lambda item: (-item[1], item[0])):
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
