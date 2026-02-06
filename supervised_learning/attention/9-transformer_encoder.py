import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
	"""
	Encodeur Transformer composé de N EncoderBlock.
	"""

	def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
		super().__init__()
		self.N = N
		self.dm = dm
		self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
		self.positional_encoding = positional_encoding(max_seq_len, dm)
		self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
		self.dropout = tf.keras.layers.Dropout(drop_rate)

	def call(self, x, training, mask):
		"""
		x: soit des indices (batch, input_seq_len) soit déjà des embeddings
		training: bool
		mask: masque pour la MHA
		Retour: (batch, input_seq_len, dm)
		"""
		# Si x est déjà en embeddings (rank 3), on l'utilise directement, sinon on embed
		if x.shape.rank == 3:
			x_embed = x
		else:
			x_embed = self.embedding(x) * tf.math.sqrt(tf.cast(self.dm, tf.float32))

		seq_len = tf.shape(x_embed)[1]
		pos = tf.cast(self.positional_encoding[:seq_len, :], x_embed.dtype)
		# ajout de l'encodage positionnel (broadcast sur la dimension batch)
		x_embed = x_embed + pos
		x_embed = self.dropout(x_embed, training=training)

		out = x_embed
		for block in self.blocks:
			out = block(out, training, mask)

		return out
