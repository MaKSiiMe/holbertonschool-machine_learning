#!/usr/bin/env python3
"""Update topics of schools by name."""


def update_topics(mongo_collection, name, topics):
	"""Update the topics field for all documents matching name."""
	mongo_collection.update_many({"name": name}, {"$set": {"topics": topics}})
