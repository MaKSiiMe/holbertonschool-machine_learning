#!/usr/bin/env python3
"""
Print the location of a GitHub user.
"""

import sys
import time
from math import ceil

import requests


def main():
    """Fetch and print the user's location from a GitHub API URL."""
    response = requests.get(sys.argv[1])

    if response.status_code == 404:
        print("Not found")
        return

    if response.status_code == 403:
        reset = int(response.headers.get("X-Ratelimit-Reset", 0))
        minutes = ceil((reset - time.time()) / 60)
        print(f"Reset in {minutes} min")
        return

    print(response.json().get("location", "Not found"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    main()
