#!/usr/bin/env python3
"""Display the first upcoming SpaceX launch with related details."""

import requests


def main():
    """Fetch and print the first upcoming launch in the required format."""
    launches = requests.get(
        "https://api.spacexdata.com/v4/launches/upcoming", timeout=10
    ).json()

    first_launch = min(launches, key=lambda launch: launch.get("date_unix", 0))

    rocket = requests.get(
        f"https://api.spacexdata.com/v4/rockets/{first_launch.get('rocket')}",
        timeout=10,
    ).json()

    launchpad_id = first_launch.get("launchpad")
    launchpad = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}",
        timeout=10,
    ).json()

    print(
        f"{first_launch.get('name')} ({first_launch.get('date_local')}) "
        f"{rocket.get('name')} - {launchpad.get('name')} "
        f"({launchpad.get('locality')})"
    )


if __name__ == "__main__":
    main()
