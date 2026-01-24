#!/usr/bin/env python3
""" Bag-of-words embedding implementation."""
import re
from collections import Counter
import numpy as np

def bag_of_words(sentences, vocab=None):
	"""
	Create a bag-of-words embedding matrix.

	Arguments:
	- sentences: list of sentences (strings)
	- vocab: optional list of features (words). If None, use all words found.

	Returns:
	- embeddings: numpy.ndarray of shape (s, f)
	- features: list of features used (lowercased)
	"""
	# Tokenize sentences to lowercase words
	if vocab is None:
		# collect all unique words across sentences
		features = sorted({
			w for s in sentences for w in re.findall(r'\w+', s.lower())
		})
	else:
		# normalize provided vocab to lowercase, keep order
		features = [w.lower() for w in vocab]

	s = len(sentences)
	f = len(features)

	# build embedding matrix of counts
	embeddings = np.zeros((s, f), dtype=int)

	for i, snt in enumerate(sentences):
		words = re.findall(r'\w+', snt.lower())
		counts = Counter(words)
		for j, feat in enumerate(features):
			embeddings[i, j] = counts.get(feat, 0)

	return embeddings, features
