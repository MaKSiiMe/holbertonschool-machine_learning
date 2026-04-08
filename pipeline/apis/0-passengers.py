#!/usr/bin/env python3
"""
Fetch ships that can carry a given number of passengers from Swapi.
"""

import requests


def availableShips(passengerCount):
    """Return ship names that can carry at least passengerCount passengers."""
    url = "https://swapi-api.hbtn.io/api/starships/"
    available = []

    while url:
        response = requests.get(url, timeout=10)
        data = response.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "").replace(",", "")
            try:
                if int(passengers) >= passengerCount:
                    available.append(ship.get("name"))
            except ValueError:
                continue

        url = data.get("next")

    return available
