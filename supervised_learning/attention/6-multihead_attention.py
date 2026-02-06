#!/usr/bin/env python3
"""Multi-head attention pour transformer"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention (transformer).
    """

    def __init__(self, dm, h):
        """
        dm: dimension du modèle, h: nombre de têtes (dm divisible par h)
        """
        super().__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x):
        """
        Convertit x de (batch, seq_len, dm) en (batch, h, seq_len, depth)
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Q: (batch, seq_len_q, dk),
        K: (batch, seq_len_v, dk),
        V: (batch, seq_len_v, dv)
        mask: broadcastable ou None
        Returns: output (batch, seq_len_q, dm),
                 weights (batch, h, seq_len_q, seq_len_v)
        """
        q = self.Wq(Q)  # (batch, seq_len_q, dm)
        k = self.Wk(K)  # (batch, seq_len_v, dm)
        v = self.Wv(V)  # (batch, seq_len_v, dm)

        q = self.split_heads(q)  # (batch, h, seq_len_q, depth)
        k = self.split_heads(k)  # (batch, h, seq_len_v, depth)
        v = self.split_heads(v)  # (batch, h, seq_len_v, depth)

        scaled_attention, weights = sdp_attention(q, k, v, mask)
        # scaled_attention: (batch, h, seq_len_q, depth)
        # rassembler les têtes -> (batch, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        batch_size = tf.shape(scaled_attention)[0]
        seq_len_q = tf.shape(scaled_attention)[1]
        # concatener têtes -> (batch, seq_len_q, dm)
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, seq_len_q, self.dm))

        output = self.linear(concat_attention)  # (batch, seq_len_q, dm)
        return output, weights
