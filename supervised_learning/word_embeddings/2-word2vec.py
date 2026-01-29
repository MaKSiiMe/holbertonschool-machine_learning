#!/usr/bin/env python3
""" Word2Vec embedding implementation."""
import re
from typing import List, Union

try:
    from gensim.models import Word2Vec
except Exception:
    Word2Vec = None


def word2vec_model(sentences: List[Union[str, List[str]]],
                   vector_size: int = 100, min_count: int = 5,
                   window: int = 5, negative: int = 5,
                   cbow: bool = True, epochs: int = 5,
                   seed: int = 0, workers: int = 1):
    """
    Crée, construit et entraîne un modèle gensim Word2Vec.

    Args:
    - sentences: list de phrases (str) ou list de tokens (list[str])
    - vector_size: dimension des embeddings
    - min_count: min occurrences pour garder un mot
    - window: fenêtre contextuelle
    - negative: nombre d'échantillons négatifs
    - cbow: True => CBOW, False => Skip-gram
    - epochs: nombre d'époques d'entraînement
    - seed: graine aléatoire
    - workers: threads pour l'entraînement

    Returns:
    - modèle Word2Vec entraîné, ou None en cas d'erreur / si gensim absent
    """
    if Word2Vec is None:
        return None
    if not isinstance(sentences, list) or len(sentences) == 0:
        return None

    # Tokenisation: accepte chaînes ou listes de tokens
    tokenized = []
    for s in sentences:
        if isinstance(s, str):
            toks = re.findall(r'\w+', s.lower())
            tokenized.append(toks)
        elif isinstance(s, (list, tuple)):
            # normalise en str lowercase
            toks = [str(t).lower() for t in s if str(t)]
            tokenized.append(toks)
        else:
            return None

    # enlever phrases vides (évite corpus vide)
    tokenized = [t for t in tokenized if t]
    if len(tokenized) == 0:
        return None

    # Détermine le mode d'entraînement (sg=0 pour CBOW, sg=1 pour skip-gram)
    sg = 0 if cbow else 1

    try:
        model = Word2Vec(vector_size=vector_size, window=window,
                         min_count=min_count, negative=negative,
                         sg=sg, seed=seed, workers=workers)
        model.build_vocab(tokenized)
        model.train(tokenized, total_examples=model.corpus_count, epochs=epochs)
    except Exception:
        return None

    return model
