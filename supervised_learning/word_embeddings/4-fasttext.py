#!/usr/bin/env python3
""" FastText embedding implementation."""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5,
                   negative=5, window=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """Crée, construit et entraîne un modèle gensim FastText.

    Args:
        sentences: list de phrases (str) ou list de tokens (list[str])
        vector_size: dimension des embeddings (int > 0)
        min_count: min occurrences pour garder un mot (int >= 0)
        negative: nombre d'échantillons négatifs (int >= 0)
        window: fenêtre contextuelle (int >= 0)
        cbow: True => CBOW, False => Skip-gram
        epochs: nombre d'époques d'entraînement (int >= 0)
        seed: graine aléatoire (int)
        workers: threads pour l'entraînement (int >= 1)

    Returns:
        Le modèle FastText entraîné, ou None en cas d'erreur.
        Si les paramètres sont invalides, retourne None.
    """
    # gensim is required; use gensim.models.FastText below
    if not isinstance(sentences, list) or len(sentences) == 0:
        return None

    # validate numeric parameters
    if not (isinstance(vector_size, int) and vector_size > 0):
        return None
    for name, val, min_allowed in (
        ("min_count", min_count, 0),
        ("window", window, 0),
        ("negative", negative, 0),
        ("epochs", epochs, 0),
        ("seed", seed, -2**63),
        ("workers", workers, 1),
    ):
        if not isinstance(val, int) or val < min_allowed:
            return None

    # Tokenisation: accepte chaînes ou listes de tokens (sans re)
    tokenized = []
    for s in sentences:
        if isinstance(s, str):
            low = s.lower()
            cleaned = ''.join(ch if ch.isalnum() else ' ' for ch in low)
            toks = cleaned.split()
            tokenized.append(toks)
        elif isinstance(s, (list, tuple)):
            toks = [str(t).lower() for t in s if str(t)]
            tokenized.append(toks)
        else:
            return None

    # enlever phrases vides (évite corpus vide)
    tokenized = [t for t in tokenized if t]
    if len(tokenized) == 0:
        return None

    sg = 0 if cbow else 1

    FastText = gensim.models.FastText

    try:
        model = FastText(vector_size=vector_size, window=window,
                         min_count=min_count, negative=negative,
                         sg=sg, seed=seed, workers=workers)
        model.build_vocab(tokenized)
        model.train(
            tokenized,
            total_examples=model.corpus_count,
            epochs=epochs,
        )
    except Exception:
        return None

    return model
