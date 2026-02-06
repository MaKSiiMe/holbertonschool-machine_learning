#!/usr/bin/env python3
"""Bloc décodeur Transformer"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Bloc décodeur Transformer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        dm: dimension du modèle
        h: nombre de têtes
        hidden: unités cachées du FFN
        drop_rate: taux de dropout
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        x: (batch, target_seq_len, dm)
        encoder_output: (batch, input_seq_len, dm)
        training: bool
        look_ahead_mask: masque pour la première MHA
        padding_mask: masque pour la deuxième MHA
        Returns: (batch, target_seq_len, dm)
        """
        # Première attention (self-attention)
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # Deuxième attention (encoder-decoder attention)
        attn2, _ = self.mha2(out1, encoder_output,
                             encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Feed-forward
        ff = self.dense_hidden(out2)
        ff = self.dense_output(ff)
        ff = self.dropout3(ff, training=training)
        out3 = self.layernorm3(out2 + ff)

        return out3
