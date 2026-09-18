"""Small pure helpers shared across the tool (easy to unit-test in isolation)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

__all__ = ["trim", "split_list", "dedupe", "strip_item", "package_name_for"]

_SEPARATORS = re.compile(r"[\s,]+")


def trim(value: str) -> str:
    return value.strip()


def split_list(value: str) -> list[str]:
    if not value:
        return []
    return [token for token in (t.strip() for t in _SEPARATORS.split(value)) if token]


def dedupe(items: Iterable[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def strip_item(items: Sequence[str], remove: str) -> list[str]:
    target = remove.lower()
    return [item for item in items if item.lower() != target]


def package_name_for(project_name: str) -> str:
    return project_name.replace("-", "_").lower()
