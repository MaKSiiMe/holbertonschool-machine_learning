#!/usr/bin/env python3
"""Insert a document in a collection using kwargs."""


def insert_school(mongo_collection, **kwargs):
	"""Insert a document in mongo_collection and return its _id."""
	return mongo_collection.insert_one(kwargs).inserted_id
