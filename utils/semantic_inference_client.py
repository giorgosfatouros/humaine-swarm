"""HTTP client for the Smart Manufacturing semantic-inference service (JSI)."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import requests

SEMANTIC_INFERENCE_BASE_URL = os.environ.get(
    "SEMANTIC_INFERENCE_BASE_URL", "http://atena.ijs.si:5008"
).rstrip("/")
SEMANTIC_INFERENCE_API_KEY = os.environ.get("SEMANTIC_INFERENCE_API_KEY")
SEMANTIC_INFERENCE_TIMEOUT = int(os.environ.get("SEMANTIC_INFERENCE_TIMEOUT", "30"))

MAX_RESULTS = 50
DEFAULT_RESULTS = 10
DEFAULT_MAX_RELATIONSHIPS = 20

SCOPE_HINTS = {
    "supported": (
        "Machine specs, availability, energy consumption, manufacturers, and "
        "capability types (Drilling, Sawing, CircleCutting) represented in the graph."
    ),
    "out_of_scope": (
        "Maintenance history, cost data, predictive failure forecasts, other HumAIne "
        "pilots, and general HumAIne platform documentation."
    ),
    "entity_types": [
        "Asset",
        "Availability",
        "CircleCutting",
        "Drilling",
        "EnergyConsumption",
        "Manufacturer",
        "Sawing",
    ],
    "cite_names_not_ids": (
        "Use human-readable entity names (e.g. Machine #271, EcoDrive elite) in "
        "user-facing answers. Keep raw Neo4j element IDs internal for follow-up lookups."
    ),
}


def _headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if SEMANTIC_INFERENCE_API_KEY:
        headers["Authorization"] = f"Bearer {SEMANTIC_INFERENCE_API_KEY}"
    return headers


def _success(**payload: Any) -> Dict[str, Any]:
    return {"success": True, "scope_hints": SCOPE_HINTS, **payload}


def _error(message: str, **extra: Any) -> Dict[str, Any]:
    return {"success": False, "error": message, "scope_hints": SCOPE_HINTS, **extra}


def _parse_error_body(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            message = err.get("message") or err.get("code")
            if message:
                return str(message)
        if body.get("message"):
            return str(body["message"])
    return response.text or f"HTTP {response.status_code}"


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{SEMANTIC_INFERENCE_BASE_URL}{path}"
    response = requests.post(
        url,
        json=payload,
        headers=_headers(),
        timeout=SEMANTIC_INFERENCE_TIMEOUT,
    )
    if not response.ok:
        raise requests.HTTPError(_parse_error_body(response), response=response)
    data = response.json()
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        err = data["error"]
        message = err.get("message") or err.get("code") or "Semantic inference error"
        raise requests.HTTPError(str(message), response=response)
    if not isinstance(data, dict):
        raise ValueError("Unexpected semantic inference response shape.")
    return data


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{SEMANTIC_INFERENCE_BASE_URL}{path}"
    response = requests.get(
        url,
        params=params,
        headers=_headers(),
        timeout=SEMANTIC_INFERENCE_TIMEOUT,
    )
    if not response.ok:
        raise requests.HTTPError(_parse_error_body(response), response=response)
    data = response.json()
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        err = data["error"]
        message = err.get("message") or err.get("code") or "Semantic inference error"
        raise requests.HTTPError(str(message), response=response)
    if not isinstance(data, dict):
        raise ValueError("Unexpected semantic inference response shape.")
    return data


async def _apost(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return await asyncio.to_thread(_post, path, payload)


async def _aget(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return await asyncio.to_thread(_get, path, params)


def _normalize_search_response(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query": data.get("query"),
        "meta": data.get("meta") if isinstance(data.get("meta"), dict) else {},
        "results": data.get("results") if isinstance(data.get("results"), list) else [],
    }


def _normalize_entity_response(data: Dict[str, Any], entity_id: str) -> Dict[str, Any]:
    if "results" in data:
        normalized = _normalize_search_response(data)
        normalized["entity_id"] = entity_id
        return normalized

    return {
        "entity_id": entity_id,
        "query": None,
        "meta": {},
        "results": [data],
    }


async def semantic_search(
    query: str,
    n: int = DEFAULT_RESULTS,
    entity_types: Optional[List[str]] = None,
    min_score: Optional[float] = None,
    include_relationships: bool = True,
    include_properties: bool = True,
    max_relationships: int = DEFAULT_MAX_RELATIONSHIPS,
) -> Dict[str, Any]:
    if not query or not query.strip():
        return _error("query is required for semantic search.")

    capped_n = max(1, min(int(n), MAX_RESULTS))
    payload: Dict[str, Any] = {
        "query": query.strip(),
        "n": capped_n,
        "include_relationships": include_relationships,
        "include_properties": include_properties,
        "max_relationships": max_relationships,
    }
    if entity_types:
        payload["entity_types"] = entity_types
    if min_score is not None:
        payload["min_score"] = min_score

    try:
        data = await _apost("/semantic-search/results", payload)
    except requests.Timeout:
        return _error(
            f"Semantic inference request timed out after {SEMANTIC_INFERENCE_TIMEOUT}s."
        )
    except requests.RequestException as exc:
        return _error(f"Semantic inference request failed: {exc}")

    normalized = _normalize_search_response(data)
    return _success(action="semantic_search", **normalized)


async def get_entity(
    entity_id: str,
    max_relationships: int = DEFAULT_MAX_RELATIONSHIPS,
) -> Dict[str, Any]:
    if not entity_id or not entity_id.strip():
        return _error("entity_id is required for entity lookup.")

    entity_id = entity_id.strip()
    params = {"max_relationships": max_relationships}

    try:
        data = await _aget(f"/entity/{entity_id}", params)
    except requests.Timeout:
        return _error(
            f"Semantic inference request timed out after {SEMANTIC_INFERENCE_TIMEOUT}s."
        )
    except requests.RequestException as exc:
        return _error(f"Semantic inference request failed: {exc}")

    normalized = _normalize_entity_response(data, entity_id)
    return _success(action="get_entity", **normalized)


async def query_manufacturing_knowledge(
    query: Optional[str] = None,
    entity_id: Optional[str] = None,
    n: int = DEFAULT_RESULTS,
    entity_types: Optional[List[str]] = None,
    min_score: Optional[float] = None,
    include_relationships: bool = True,
    include_properties: bool = True,
    max_relationships: int = DEFAULT_MAX_RELATIONSHIPS,
) -> Dict[str, Any]:
    if entity_id:
        return await get_entity(
            entity_id=entity_id,
            max_relationships=max_relationships,
        )
    if not query:
        return _error("Either query or entity_id is required.")
    return await semantic_search(
        query=query,
        n=n,
        entity_types=entity_types,
        min_score=min_score,
        include_relationships=include_relationships,
        include_properties=include_properties,
        max_relationships=max_relationships,
    )
