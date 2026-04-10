#!/usr/bin/env python3
"""Find schools by topic."""


def schools_by_topic(mongo_collection, topic):
	"""Return all schools that contain the given topic."""
	return mongo_collection.find({"topics": topic})
