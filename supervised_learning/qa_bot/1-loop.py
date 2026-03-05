#!/usr/bin/env python3
"""Interactive QA loop.

Prompts with `Q: ` and prints responses prefixed with `A:`.
If the user types `exit`, `quit`, `goodbye` or `bye` (case-insensitive)
prints `A: Goodbye` and exits.

Uses `question_answer` from `0-qa.py` and concatenates all markdown
articles in the `ZendeskArticles/` folder as the reference context.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

question_answer = __import__("0-qa").question_answer

EXIT_WORDS = {"exit", "quit", "goodbye", "bye"}


def _load_reference() -> str:
    base = Path(__file__).resolve().parent
    docs_dir = base / "ZendeskArticles"
    if not docs_dir.is_dir():
        return ""
    parts = []
    for p in sorted(docs_dir.glob("*.md")):
        try:
            parts.append(p.read_text(encoding="utf-8"))
        except Exception:
            # ignore unreadable files
            continue
    return "\n\n".join(parts)


def _user_wants_exit(text: str) -> bool:
    return text.strip().lower() in EXIT_WORDS


def main() -> None:
    reference = _load_reference()

    try:
        while True:
            try:
                q = input("Q: ")
            except EOFError:
                break

            if _user_wants_exit(q):
                print("A: Goodbye")
                break

            # obtain answer (may be None)
            ans: Optional[str] = None
            try:
                ans = question_answer(q, reference)
            except Exception:
                # quietly handle model/tokenizer errors in interactive loop
                ans = None

            print(f"A: {ans}" if ans is not None else "A: None")
    except KeyboardInterrupt:
        # clean exit on Ctrl-C
        print()


if __name__ == "__main__":
    main()
