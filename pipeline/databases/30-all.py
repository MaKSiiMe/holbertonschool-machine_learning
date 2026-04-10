#!/usr/bin/env python3
"""List all documents in a collection."""


def list_all(mongo_collection):
	"""Return all documents in mongo_collection as a list."""
	return list(mongo_collection.find())
