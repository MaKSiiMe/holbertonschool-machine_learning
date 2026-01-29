#!/usr/bin/env python3
"""N-gram BLEU score."""
import math


def _ngrams(sequence, n):
    """Yield n-grams as tuples from a sequence of tokens."""
    if n <= 0:
        return []
    L = len(sequence)
    if L < n:
        return []
    return [tuple(sequence[i:i + n]) for i in range(L - n + 1)]


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence.

    references: list of reference translations (each a list of words)
    sentence: list of words for the candidate sentence
    n: size of n-gram

    Returns: n-gram BLEU score (float)
    """
    # empty candidate or invalid n
    if not sentence or n <= 0:
        return 0.0

    # candidate length
    c = len(sentence)

    # choose reference length r as the reference with length closest to c
    ref_lens = [len(r) for r in references]
    closest = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))
    r = closest

    # build candidate n-grams counts
    cand_ngrams = _ngrams(sentence, n)
    total_cand = len(cand_ngrams)
    if total_cand == 0:
        return 0.0

    cand_counts = {}
    for ng in cand_ngrams:
        cand_counts[ng] = cand_counts.get(ng, 0) + 1

    # compute max reference counts for each n-gram
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = _ngrams(ref, n)
        ref_count = {}
        for ng in ref_ngrams:
            ref_count[ng] = ref_count.get(ng, 0) + 1
        for ng, cnt in ref_count.items():
            if cnt > max_ref_counts.get(ng, 0):
                max_ref_counts[ng] = cnt

    # clipped count
    clipped = 0
    for ng, cnt in cand_counts.items():
        clipped += min(cnt, max_ref_counts.get(ng, 0))

    # precision
    p_n = clipped / total_cand

    # brevity penalty
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1 - (r / c))

    return bp * p_n
