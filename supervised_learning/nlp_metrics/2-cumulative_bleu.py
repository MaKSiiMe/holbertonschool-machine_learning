#!/usr/bin/env python3
"""Cumulative n-gram BLEU score."""
import math


def _ngrams(sequence, n):
    """Return a list of n-grams for a sequence.

    Args:
        sequence: list of tokens (words).
        n: size of the n-gram (int).

    Returns:
        A list of n-grams as tuples. If `n` <= 0 or the sequence is shorter
        than `n`, an empty list is returned.
    """
    if n <= 0:
        return []
    L = len(sequence)
    if L < n:
        return []
    return [tuple(sequence[i:i + n]) for i in range(L - n + 1)]


def cumulative_bleu(references, sentence, n):
    """Calculates the cumulative n-gram BLEU score for a sentence.

    references: list of reference translations (each a list of words)
    sentence: list of words for the candidate sentence
    n: largest n-gram size to use

    Returns: cumulative BLEU score (float)
    """
    if not sentence or n <= 0:
        return 0.0

    c = len(sentence)
    ref_lens = [len(r) for r in references]
    # choose reference length closest to c (tie -> shortest)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    precisions = []
    for k in range(1, n + 1):
        cand_ngrams = _ngrams(sentence, k)
        total_cand = len(cand_ngrams)
        if total_cand == 0:
            return 0.0

        cand_counts = {}
        for ng in cand_ngrams:
            cand_counts[ng] = cand_counts.get(ng, 0) + 1

        # max reference counts
        max_ref_counts = {}
        for ref in references:
            ref_ngrams = _ngrams(ref, k)
            ref_count = {}
            for ng in ref_ngrams:
                ref_count[ng] = ref_count.get(ng, 0) + 1
            for ng, cnt in ref_count.items():
                if cnt > max_ref_counts.get(ng, 0):
                    max_ref_counts[ng] = cnt

        clipped = 0
        for ng, cnt in cand_counts.items():
            clipped += min(cnt, max_ref_counts.get(ng, 0))

        p_k = clipped / total_cand
        precisions.append(p_k)

    # if any precision is zero, geometric mean is zero
    if any(p == 0 for p in precisions):
        return 0.0

    # geometric mean of precisions (equal weights)
    log_sum = sum(math.log(p) for p in precisions)
    geo_mean = math.exp(log_sum / len(precisions))

    # brevity penalty
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1 - (r / c))

    return bp * geo_mean
