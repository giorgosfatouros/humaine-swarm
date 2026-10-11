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
    re.compile(r"\bhaic benchmark\b", re.I),
    re.compile(r"\bstored on haic\b", re.I),
    re.compile(r"\bcompare my haic\b", re.I),
]

# Documentation-style HAIC questions (reference tables, "how many metrics are listed",
# definitions). These are answered from the docs via get_docs, never from the live API.
_DOC_PATTERNS = [
    re.compile(r"\b(reference|documentation|docs?|guide|overview|glossary)\b", re.I),
    re.compile(r"\bsummary table\b", re.I),
    re.compile(r"\blisted in\b", re.I),
    re.compile(r"\bhow many\b.*\bmetrics?\b", re.I),
    re.compile(r"\b(core|main|available|supported)\s+(haic\s+)?metrics\b", re.I),
]

# Messages that ask for an action beyond reading stored results, or that name another
# pilot's domain. The pre-router must not answer these with the user's stored scores;
# they fall through to the LLM, which can decline or pick the right tool.
_ACTION_PATTERNS = [
    re.compile(r"\b(deploy|launch|execute|run|create|start|train|delete|upload|trigger)\b", re.I),
]
_OTHER_DOMAIN_PATTERN = re.compile(
    r"\b(manufactur\w*|drilling|energy|oncolog\w*|healthcare|diabetes|ticketing)\b", re.I
)

# Broad metric-name patterns count as a live-data request only when the user says "my"/"our pilot";
# without that, "Trust Proxy metric in HAIC" is a conceptual question for get_docs.
_LIVE_PATTERNS_POSSESSIVE = [
    re.compile(r"\bhaic\b.*\b(trust|hcl|metrics?)\b", re.I),
    re.compile(r"\b(trust|hcl)\b.*\bhaic\b", re.I),
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

    if any(p.search(normalized) for p in _ACTION_PATTERNS) or _OTHER_DOMAIN_PATTERN.search(normalized):
        return None

    if not has_my and any(p.search(normalized) for p in _DOC_PATTERNS):
        return None

    for pattern in _CONCEPTUAL_PATTERNS:
        if pattern.search(normalized) and not has_my:
            return None

    for pattern in _LIVE_PATTERNS:
        if pattern.search(normalized):
            return infer_haic_action(normalized)

    if has_my and any(p.search(normalized) for p in _LIVE_PATTERNS_POSSESSIVE):
        return infer_haic_action(normalized)

    if has_my and re.search(r"\bhaic\b", lower, re.I):
        return infer_haic_action(normalized)

    return None
