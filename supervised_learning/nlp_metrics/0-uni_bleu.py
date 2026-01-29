#!/usr/bin/env python3
"""Unigram BLEU score."""
import math


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence.

    references: list of reference translations (each a list of words)
    sentence: list of words for the candidate sentence

    Returns: unigram BLEU score (float)
    """
    # empty candidate
    if not sentence:
        return 0.0

    # candidate length
    c = len(sentence)

    # choose reference length r as the reference with length closest to c
    ref_lens = [len(r) for r in references]
    # pick closest; if tie, choose the shortest (standard BLEU tie-break)
    closest = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))
    r = closest

    # count unigrams in candidate
    cand_counts = {}
    for w in sentence:
        cand_counts[w] = cand_counts.get(w, 0) + 1

    # compute max reference counts for each word
    max_ref_counts = {}
    for ref in references:
        ref_count = {}
        for w in ref:
            ref_count[w] = ref_count.get(w, 0) + 1
        for w, cnt in ref_count.items():
            if cnt > max_ref_counts.get(w, 0):
                max_ref_counts[w] = cnt

    # clipped count
    clipped = 0
    for w, cnt in cand_counts.items():
        clipped += min(cnt, max_ref_counts.get(w, 0))

    # precision
    p1 = clipped / c

    # brevity penalty
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1 - (r / c))

    return bp * p1
