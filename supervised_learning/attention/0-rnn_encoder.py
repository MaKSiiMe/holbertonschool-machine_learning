#!/usr/bin/env python3
"""RNN encoder for machine translation"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN encoder using an embedding layer and a GRU.

    Public instance attributes:
        batch: the batch size
        units: the number of hidden units in the RNN cell
        embedding: a keras Embedding layer
        gru: a keras GRU layer
    """

    def __init__(self, vocab, embedding, units, batch):
        """Class constructor.

        Args:
            vocab: int, size of the input vocabulary
            embedding: int, dimensionality of the embedding vector
            units: int, number of hidden units in the RNN cell
            batch: int, batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """Initializes hidden states for the GRU to zeros.

        Returns:
            tf.Tensor of shape (batch, units)
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Runs the encoder on input x starting from initial state.

        Args:
            x: tf.Tensor of shape (batch, input_seq_len) of word indices
            initial: tf.Tensor of shape (batch, units) for initial hidden state

        Returns:
            outputs: tf.Tensor of shape (batch, input_seq_len, units)
            hidden: tf.Tensor of shape (batch, units)
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
