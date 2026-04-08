#!/usr/bin/env python3
"""
Fetch the home planets of sentient species from Swapi.
"""

import requests


def sentientPlanets():
    """Return the list of home planet names for sentient species."""
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url:
        response = requests.get(url, timeout=10)
        data = response.json()

        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            if (
                "sentient" not in classification
                and "sentient" not in designation
            ):
                continue

            homeworld = species.get("homeworld")
            if not homeworld:
                continue

            homeworld_response = requests.get(homeworld, timeout=10)
            planet_name = homeworld_response.json().get("name")
            if planet_name:
                planets.append(planet_name)

        url = data.get("next")

    return planets
