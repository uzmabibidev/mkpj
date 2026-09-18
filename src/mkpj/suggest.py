"""'Did you mean …' suggestions for mistyped package/extension names."""

from __future__ import annotations

import difflib
from collections.abc import Sequence

__all__ = ["fuzzy_suggest", "format_suggestions"]


def fuzzy_suggest(
    query: str, candidates: Sequence[str], n: int = 3, cutoff: float = 0.5
) -> list[str]:
    if not query:
        return []
    return difflib.get_close_matches(query, list(candidates), n=n, cutoff=cutoff)


def format_suggestions(matches: Sequence[str]) -> str:
    return " ".join(matches)
