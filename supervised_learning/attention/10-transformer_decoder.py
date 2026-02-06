import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
	"""
	Décodeur Transformer composé de N DecoderBlock.
	"""

	def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
		super().__init__()
		self.N = N
		self.dm = dm
		self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
		self.positional_encoding = positional_encoding(max_seq_len, dm)
		self.blocks = [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
		self.dropout = tf.keras.layers.Dropout(drop_rate)

	def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
		"""
		x: soit des indices (batch, target_seq_len) soit déjà des embeddings
		encoder_output: (batch, input_seq_len, dm)
		training: bool
		look_ahead_mask, padding_mask: masques pour les MHA
		Retour: (batch, target_seq_len, dm)
		"""
		if x.shape.rank == 3:
			x_embed = x
		else:
			x_embed = self.embedding(x) * tf.math.sqrt(tf.cast(self.dm, tf.float32))

		seq_len = tf.shape(x_embed)[1]
		pos = tf.cast(self.positional_encoding[:seq_len, :], x_embed.dtype)
		x_embed = x_embed + pos
		x_embed = self.dropout(x_embed, training=training)

		out = x_embed
		for block in self.blocks:
			out = block(out, encoder_output, training, look_ahead_mask, padding_mask)

		return out
