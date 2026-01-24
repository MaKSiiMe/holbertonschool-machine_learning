#!/usr/bin/env python3
""" TF-IDF embedding implementation."""
import re
from collections import Counter
import numpy as np

def tf_idf(sentences, vocab=None):
	"""
	Create a TF-IDF embedding matrix.

	Arguments:
	- sentences: list of sentences (strings)
	- vocab: optional list of features (words). If None, use all words found.

	Returns:
	- embeddings: numpy.ndarray of shape (s, f)
	- features: list of features used (lowercased)
	"""
	# Tokenize and decide features
	if vocab is None:
		features = sorted({
			w for s in sentences for w in re.findall(r'\w+', s.lower())
		})
	else:
		features = [w.lower() for w in vocab]

	s = len(sentences)
	f = len(features)

	# precompute document frequencies (df)
	df_counts = np.zeros(f, dtype=int)
	tokenized_sentences = []
	for snt in sentences:
		words = re.findall(r'\w+', snt.lower())
		tokenized_sentences.append(words)
		unique_words = set(words)
		for j, feat in enumerate(features):
			if feat in unique_words:
				df_counts[j] += 1

	idf = np.log((s + 1) / (df_counts + 1)) + 1.0

	# build TF-IDF matrix
	embeddings = np.zeros((s, f), dtype=float)
	for i, words in enumerate(tokenized_sentences):
		total = len(words)
		if total == 0:
			continue
		counts = Counter(words)
		for j, feat in enumerate(features):
			tf = counts.get(feat, 0) / total
			embeddings[i, j] = tf * idf[j]

	return embeddings, features
