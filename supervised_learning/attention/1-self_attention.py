#!/usr/bin/env python3
"""Self-attention mechanism for machine translation"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
	"""
	Attention de type Bahdanau pour traduction automatique.
	"""

	def __init__(self, units):
		"""units: nombre d'unités cachées dans le modèle d'alignement"""
		super().__init__()
		self.W = tf.keras.layers.Dense(units)
		self.U = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, s_prev, hidden_states):
		"""
		s_prev: (batch, units) - état caché précédent du décodeur
		hidden_states: (batch, input_seq_len, units) - sorties de l'encodeur
		Returns: context (batch, units), weights (batch, input_seq_len, 1)
		"""
		# W(s_prev) -> (batch, units) puis (batch, 1, units)
		score_part = tf.expand_dims(self.W(s_prev), 1)
		# U(hidden_states) -> (batch, input_seq_len, units)
		hidden_part = self.U(hidden_states)
		# score -> (batch, input_seq_len, 1)
		score = self.V(tf.nn.tanh(score_part + hidden_part))
		# weights -> softmax sur l'axe temporel
		weights = tf.nn.softmax(score, axis=1)
		# context -> somme pondérée des hidden states (batch, units)
		context = tf.reduce_sum(weights * hidden_states, axis=1)
		return context, weights
