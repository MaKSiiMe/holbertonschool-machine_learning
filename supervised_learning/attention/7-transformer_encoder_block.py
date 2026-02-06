import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
	"""
	Bloc d'encodeur Transformer.
	"""

	def __init__(self, dm, h, hidden, drop_rate=0.1):
		"""
		dm: dimension du modèle
		h: nombre de têtes
		hidden: unités cachées du FFN
		drop_rate: taux de dropout
		"""
		super().__init__()
		self.mha = MultiHeadAttention(dm, h)
		self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
		self.dense_output = tf.keras.layers.Dense(dm)
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.dropout1 = tf.keras.layers.Dropout(drop_rate)
		self.dropout2 = tf.keras.layers.Dropout(drop_rate)

	def call(self, x, training, mask=None):
		"""
		x: (batch, input_seq_len, dm)
		training: bool
		mask: masque pour la MHA
		Returns: (batch, input_seq_len, dm)
		"""
		# Multi-head attention (Q=K=V=x)
		attn_output, _ = self.mha(x, x, x, mask)
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(x + attn_output)

		# Feed-forward
		ff = self.dense_hidden(out1)
		ff = self.dense_output(ff)
		ff = self.dropout2(ff, training=training)
		out2 = self.layernorm2(out1 + ff)

		return out2
