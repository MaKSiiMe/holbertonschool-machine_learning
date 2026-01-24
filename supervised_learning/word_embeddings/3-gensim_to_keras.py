#!/usr/bin/env python3
""" Convert a gensim Word2Vec/KeyedVectors model to a Keras Embedding layer."""
import numpy as np
import tensorflow as tf


def gensim_to_keras(model):
	"""
	Converts a trained gensim Word2Vec/KeyedVectors model to a Keras Embedding.

	Args:
	- model: a gensim Word2Vec model or KeyedVectors instance

	Returns:
	- tf.keras.layers.Embedding initialized with gensim vectors (trainable),
	  or None on failure.

	Note: Indices used by the Embedding correspond to the order of
	model.wv.index_to_key (or model.index_to_key for KeyedVectors).
	"""
	if model is None:
		return None

	# accept Word2Vec model or KeyedVectors directly
	kv = getattr(model, "wv", model)

	# get vectors (gensim>=4: .vectors, older: .syn0)
	vectors = getattr(kv, "vectors", None)
	if vectors is None:
		vectors = getattr(kv, "syn0", None)

	if vectors is None:
		return None

	try:
		vectors = np.array(vectors, dtype=np.float32)
		vocab_size, vector_size = vectors.shape
	except Exception:
		return None

	# Create a trainable Embedding layer initialized with gensim weights
	emb_layer = tf.keras.layers.Embedding(
		input_dim=vocab_size,
		output_dim=vector_size,
		embeddings_initializer=tf.keras.initializers.Constant(vectors),
		trainable=True
	)
	return emb_layer
