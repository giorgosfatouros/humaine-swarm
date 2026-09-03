"""Detect user messages that should route to query_haic_benchmark (not MinIO/RAG)."""

from __future__ import annotations

import re
from typing import Optional

# Conceptual HAIC questions — no live API call unless user also asks for their numbers.
_CONCEPTUAL_PATTERNS = [
    re.compile(r"\bwhat does\b.*\bmean\b", re.I),
    re.compile(r"\bwhat is\b.*\b(framework|schema|metric definition)\b", re.I),
    re.compile(r"\bhow is\b.*\b(computed|calculated)\b", re.I),
    re.compile(r"\blogging schema\b", re.I),
    re.compile(r"\binterpret(ation)?\b.*\b(metric|score)\b", re.I),
]

# Live pilot data — fetch from HAIC API.
_LIVE_PATTERNS = [
    re.compile(r"\bfind my haic\b", re.I),
    re.compile(r"\b(show|get|list)\b.*\bhaic\b", re.I),
    re.compile(r"\bmy haic\b", re.I),
    re.compile(r"\bhaic results?\b", re.I),
    re.compile(r"\bhaic scores?\b", re.I),
    re.compile(r"\bhaic evaluations?\b", re.I),
    re.compile(r"\bhaic\b.*\b(trust|hcl|metrics?)\b", re.I),
    re.compile(r"\b(trust|hcl)\b.*\bhaic\b", re.I),
    re.compile(r"\bhaic benchmark\b", re.I),
    re.compile(r"\bstored on haic\b", re.I),
    re.compile(r"\bcompare my haic\b", re.I),
]

_LIST_EVAL_PATTERNS = [
    re.compile(r"\bevaluations?\b.*\b(do i have|available)\b", re.I),
    re.compile(r"\bwhat evaluations?\b", re.I),
    re.compile(r"\blist evaluations?\b", re.I),
]


def infer_haic_action(text: str) -> str:
    """Pick the default query_haic_benchmark action for a live-data question."""
    for pattern in _LIST_EVAL_PATTERNS:
        if pattern.search(text):
            return "list_evaluations"
    return "get_holistic"


def detect_haic_live_query(text: str) -> Optional[str]:
    """
    Return a query_haic_benchmark action if the message asks for the user's
    stored HAIC results, else None.
    """
    if not text or not text.strip():
        return None

    normalized = text.strip()
    lower = normalized.lower()

    if "haic" not in lower and "benchmark suite" not in lower:
        return None

    has_my = bool(re.search(r"\b(my|mine|our pilot)\b", lower, re.I))

    for pattern in _CONCEPTUAL_PATTERNS:
        if pattern.search(normalized) and not has_my:
            return None

    for pattern in _LIVE_PATTERNS:
        if pattern.search(normalized):
            return infer_haic_action(normalized)

    if has_my and re.search(r"\bhaic\b", lower, re.I):
        return infer_haic_action(normalized)

    return None
