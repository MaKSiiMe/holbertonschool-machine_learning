#!/usr/bin/env python3
"""Scaled Dot-Product Attention pour transformer"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calcul du scaled dot-product attention.
    Args:
            Q: tensor(..., seq_len_q, dk)
            K: tensor(..., seq_len_v, dk)
            V: tensor(..., seq_len_v, dv)
            mask: broadcastable tensor(..., seq_len_q, seq_len_v) ou None
    Returns:
            output: tensor(..., seq_len_q, dv)
            weights: tensor(..., seq_len_q, seq_len_v)
    """
    # matmul Q et K^T -> (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    # scale par sqrt(dk)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_scores = matmul_qk / tf.math.sqrt(dk)
    # appliquer le mask si fourni (add mask * -1e9)
    if mask is not None:
        mask = tf.cast(mask, scaled_scores.dtype)
        scaled_scores += (mask * -1e9)
    # softmax sur l'axe des clés -> (..., seq_len_q, seq_len_v)
    weights = tf.nn.softmax(scaled_scores, axis=-1)
    # output -> (..., seq_len_q, dv)
    output = tf.matmul(weights, V)
    return output, weights
