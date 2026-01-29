#!/usr/bin/env python3
""" Convert a gensim Word2Vec/KeyedVectors model to a Keras Embedding layer."""
import numpy as np
try:
    import tensorflow as tf
except Exception:
    tf = None


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
    if model is None or tf is None:
        return None

    # Ensure tf.keras is available
    if not hasattr(tf, "keras") or not hasattr(tf.keras, "layers"):
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
        # reject invalid numeric values
        if not np.isfinite(vectors).all():
            return None
        if vectors.ndim != 2:
            return None
        vocab_size, vector_size = vectors.shape
        if vocab_size == 0 or vector_size == 0:
            return None
    except Exception:
        return None

    # Create a trainable Embedding layer initialized with gensim weights
    try:
        emb_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=vector_size,
            embeddings_initializer=tf.keras.initializers.Constant(vectors),
            trainable=True
        )
    except Exception:
        return None

    return emb_layer
