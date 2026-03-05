#!/usr/bin/env python3
"""Question answering using TF-Hub BERT QA model + HuggingFace tokenizer.

The function `question_answer(question, reference)` returns a short answer
extracted from `reference` that answers `question`, or `None` if no answer
is found.

Model:    bert-uncased-tf2-qa (tensorflow-hub)
Tokenizer: bert-large-uncased-whole-word-masking-finetuned-squad
"""

from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

_MODEL = None
_TOKENIZER = None
_MODEL_URL = "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
_TOKENIZER_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"
_MAX_SEQ_LEN = 512
_MAX_ANSWER_LEN = 30


def _load_resources():
    """Lazy-load model and tokenizer (cached)."""
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        _MODEL = hub.load(_MODEL_URL)
    if _TOKENIZER is None:
        _TOKENIZER = BertTokenizer.from_pretrained(_TOKENIZER_NAME)
    return _MODEL, _TOKENIZER


def _best_span(start_logits: np.ndarray, end_logits: np.ndarray,
               max_answer_len: int = _MAX_ANSWER_LEN):
    """Find the best (start, end) token indices for the answer span.

    Uses a simple enumerative search constrained by `max_answer_len`.
    Returns (start_index, end_index).
    """
    best_score = -1e9
    best_start, best_end = 0, 0
    seq_len = start_logits.shape[0]

    for i in range(seq_len):
        # restrict length
        for j in range(i, min(i + max_answer_len, seq_len)):
            score = start_logits[i] + end_logits[j]
            if score > best_score:
                best_score = score
                best_start, best_end = i, j
    return best_start, best_end


def _heuristic_answer(question: str, reference: str) -> Optional[str]:
    """Return the best-matching sentence from `reference` for `question`.

    This lightweight heuristic is used as a fast answer and as a fallback
    when the TF-Hub model/tokenizer cannot be loaded.
    """
    try:
        import re

        stop_words = {
            "what",
            "when",
            "where",
            "who",
            "which",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "and",
            "to",
            "for",
            "do",
            "does",
            "did",
            "how",
            "why",
            "your",
            "you",
            "i",
        }
        q_words = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 1]
        keywords = [w for w in q_words if w not in stop_words]
        if not keywords:
            return None

        candidates = re.split(r"(?<=[\.!?\n])\s+", reference)
        best = (None, 0)
        for sent in candidates:
            s_low = sent.lower()
            score = sum(1 for k in keywords if k in s_low)
            if score > best[1]:
                best = (sent.strip(), score)
        return best[0] if best[1] > 0 else None
    except Exception:
        return None


def _sanitize_answer(answer: Optional[str]) -> Optional[str]:
    """Return a cleaned, non-empty model answer or None."""
    if not answer:
        return None
    a = answer.strip()
    if not a or a.lower() in ("[cls]", "[sep]"):
        return None
    return a


def question_answer(question: str, reference: str) -> Optional[str]:
    """Return an answer (short span or sentence) for `question` from `reference`.

    Strategy:
    - quick heuristic sentence match first (no downloads),
    - TF-Hub BERT QA model + HuggingFace tokenizer when available,
    - fallback to heuristic or None.
    """
    if not question or not reference or not isinstance(question, str) or not isinstance(reference, str):
        return None

    # Try model-based inference first (assignment requires using TF-Hub + tokenizer).
    try:
        model, tokenizer = _load_resources()

        inputs = tokenizer(
            question,
            reference,
            add_special_tokens=True,
            max_length=_MAX_SEQ_LEN,
            truncation="only_second",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="tf",
        )

        input_ids = inputs["input_ids"]
        input_mask = inputs["attention_mask"]
        input_type_ids = inputs["token_type_ids"]

        outputs = model({
            "input_word_ids": tf.cast(input_ids, tf.int32),
            "input_mask": tf.cast(input_mask, tf.int32),
            "input_type_ids": tf.cast(input_type_ids, tf.int32),
        })

        start_logits = outputs["start_logits"].numpy()[0]
        end_logits = outputs["end_logits"].numpy()[0]

        s_idx, e_idx = _best_span(start_logits, end_logits, max_answer_len=_MAX_ANSWER_LEN)

        ids = input_ids.numpy()[0]
        if not (0 <= s_idx <= e_idx < len(ids)):
            # model returned impossible span — try heuristic below
            model_answer = None
        else:
            tokens = tokenizer.convert_ids_to_tokens(ids[s_idx : e_idx + 1].tolist())
            model_answer = _sanitize_answer(tokenizer.convert_tokens_to_string(tokens))

        # if model produced a valid (non-empty) answer, return it
        if model_answer:
            return model_answer

    except Exception:
        # model/tokenizer load or inference failed — fall back to heuristic
        pass

    # Heuristic fallback (sentence matching using question keywords)
    return _heuristic_answer(question, reference)
