#!/usr/bin/env python3
""" Convert a gensim Word2Vec/KeyedVectors model to a Keras Embedding layer."""
import tensorflow as tf


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
    if model is None:
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
        vectors = tf.convert_to_tensor(vectors, dtype=tf.float32)
        # reject invalid numeric values
        if not bool(tf.reduce_all(tf.math.is_finite(vectors)).numpy()):
            return None
        # if gensim provides an index order, ensure vectors follow it;
        # if lengths mismatch, try to rebuild vectors in index_to_key order
        idx_keys = getattr(kv, "index_to_key", None)
        if idx_keys is not None:
            try:
                vec_count = vectors.shape[0]
                if vec_count is None:
                    vec_count = int(tf.shape(vectors)[0].numpy())
                else:
                    vec_count = int(vec_count)
            except Exception:
                vec_count = int(tf.shape(vectors)[0].numpy())
            if len(idx_keys) != vec_count:
                try:
                    stacked = tf.stack([kv.get_vector(k) for k in idx_keys])
                    vectors = tf.cast(stacked, tf.float32)
                except Exception:
                    return None
        # ensure 2D
        if tf.rank(vectors).numpy() != 2:
            return None
        shape = tf.shape(vectors)
        vocab_size = int(shape[0].numpy())
        vector_size = int(shape[1].numpy())
        if vocab_size == 0 or vector_size == 0:
            return None
    except Exception:
        return None

    # Create a trainable Embedding layer initialized with gensim weights
    try:
        weights = vectors.numpy()
        emb_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=vector_size,
            embeddings_initializer=tf.keras.initializers.Constant(weights),
            trainable=True
        )
    except Exception:
        return None

    return emb_layer
