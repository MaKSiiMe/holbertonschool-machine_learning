#!/usr/bin/env python3
""" TF-IDF embedding implementation."""
import re
from collections import Counter
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Create a TF-IDF embedding matrix.

    Arguments:
    - sentences: list of sentences (strings)
    - vocab: optional list of features (words). If None, use all words found.

    Returns:
    - embeddings: numpy.ndarray of shape (s, f)
    - features: list of features used (lowercased)
    """
    # Tokenize and decide features
    if vocab is None:
        # collect all unique words across sentences, ignore single-char tokens
        features = sorted(
            {
                w
                for s in sentences
                for w in re.findall(r'\w+', s.lower())
                if len(w) > 1
            }
        )
    else:
        # normalize provided vocab, keep order, deduplicate
        # and ignore single-char tokens
        seen = set()
        features = []
        for w in vocab:
            lw = w.lower()
            if not re.fullmatch(r'\w+', lw) or len(lw) == 1:
                continue
            if lw not in seen:
                seen.add(lw)
                features.append(lw)

    s = len(sentences)
    f = len(features)

    # tokenize sentences and compute document frequencies efficiently
    df_counts = np.zeros(f, dtype=int)
    tokenized_sentences = []
    feat_to_idx = {feat: idx for idx, feat in enumerate(features)}
    for snt in sentences:
        # Filter tokens of length 1 so TF matches features selection
        words = [w for w in re.findall(r'\w+', snt.lower()) if len(w) > 1]
        tokenized_sentences.append(words)
        unique_words = set(words)
        for w in unique_words:
            if w in feat_to_idx:
                df_counts[feat_to_idx[w]] += 1

    idf = np.log((s + 1) / (df_counts + 1)) + 1.0

    # build TF-IDF matrix
    embeddings = np.zeros((s, f), dtype=float)
    for i, words in enumerate(tokenized_sentences):
        total = len(words)
        if total == 0:
            continue
        counts = Counter(words)
        for w, c in counts.items():
            if w in feat_to_idx:
                j = feat_to_idx[w]
                tf = c / total
                embeddings[i, j] = tf * idf[j]

    # normalize each sentence vector to unit L2 norm (avoid division by zero)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    return embeddings, features
