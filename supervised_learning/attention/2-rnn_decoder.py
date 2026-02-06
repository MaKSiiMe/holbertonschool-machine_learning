#!/usr/bin/env python3
"""RNN Decoder with Bahdanau attention for machine translation"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Decoder RNN avec attention.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        vocab: taille du vocabulaire de sortie
        embedding: dimension des embeddings
        units: nombre d'unités cachées de la GRU
        batch: taille du batch
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        x: (batch, 1) - indice du mot précédent
        s_prev: (batch, units) - état caché précédent du décodeur
        hidden_states: (batch, input_seq_len, units) - sorties de l'encodeur
        Returns: y (batch, vocab), s (batch, units)
        """
        # context: (batch, units), weights: (batch, input_seq_len, 1)
        context, weights = self.attention(s_prev, hidden_states)
        # embedding x -> (batch, 1, embedding_dim)
        x = self.embedding(x)
        # expand context pour avoir pas temporel -> (batch, 1, units)
        context = tf.expand_dims(context, 1)
        # concaténer context puis x sur la dernière dimension
        x_and_context = tf.concat([context, x], axis=-1)
        # passer dans la GRU
        outputs, state = self.gru(x_and_context)
        # prendre la sortie du dernier pas temporel -> (batch, units)
        outputs = outputs[:, -1, :]
        # projection finale -> (batch, vocab)
        y = self.F(outputs)
        return y, state
