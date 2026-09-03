"""HAIC Benchmark Suite HTTP client and compact response projectors."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import requests

from utils.haic_pilot_map import assert_configuration_allowed

HAIC_BASE_URL = os.environ.get(
    "HAIC_BASE_URL", "https://benchmark.humaine-horizon.eu/api"
).rstrip("/")
HAIC_API_KEY = os.environ.get("HAIC_API_KEY")
HAIC_REQUEST_TIMEOUT = int(os.environ.get("HAIC_REQUEST_TIMEOUT", "30"))

HAIC_METRIC_KEYS = ("F", "D", "HCL", "Tr", "A", "S", "EL", "EfficiencyScore")

INTERPRETATION_HINTS = {
    "Tr": "Trust Proxy: fraction of AI suggestions accepted by the human (0–1). Not subjective trust.",
    "HCL": "Human-Centeredness: lower cognitive friction; higher is better (0–1 scale).",
    "F": "Interaction Frequency: interactions per minute.",
    "D": "Mean Action Duration in seconds.",
    "EL": "Effort Loss: dimensionless overhead vs baseline; null if baseline or data insufficient.",
    "A": "Adaptability: early-vs-late improvement; null if session too short or unlabeled.",
    "S": "Surrogate Similarity; null if no surrogate distributions logged.",
    "EfficiencyScore": "Composite efficiency indicator when present.",
    "missing_metric": "A null metric usually means preconditions were not met, not a system failure.",
    "quadrants": "Low EL + high Tr = efficient and trusted; high HCL + high F = frequent smooth interaction.",
}


def _headers() -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if HAIC_API_KEY:
        headers["Authorization"] = f"Bearer {HAIC_API_KEY}"
    return headers


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{HAIC_BASE_URL}{path}"
    response = requests.get(
        url, params=params, headers=_headers(), timeout=HAIC_REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


async def _aget(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    return await asyncio.to_thread(_get, path, params)


def _dedupe_warnings(warnings: Any, limit: int = 20) -> List[Dict[str, str]]:
    if not isinstance(warnings, list):
        return []
    seen: set[tuple[str, str]] = set()
    out: List[Dict[str, str]] = []
    for item in warnings:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric", ""))
        text = str(item.get("warning", ""))
        key = (metric, text)
        if key in seen:
            continue
        seen.add(key)
        out.append({"metric": metric, "warning": text})
        if len(out) >= limit:
            break
    return out


def _extract_haic_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("haic"), dict):
        return {k: payload["haic"].get(k) for k in HAIC_METRIC_KEYS if k in payload["haic"]}
    interaction = (payload.get("aggregates") or {}).get("interaction")
    if isinstance(interaction, dict):
        return {k: interaction.get(k) for k in HAIC_METRIC_KEYS if k in interaction}
    return {}


def _project_configuration(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "application_name": row.get("application_name"),
        "ai_model_name": row.get("ai_model_name"),
        "pilot_tag": row.get("pilot_tag"),
        "evaluation_status": row.get("evaluation_status"),
        "evaluation_date": row.get("evaluation_date"),
        "description": row.get("description"),
    }


def _project_result_index(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "ai_model_version": row.get("ai_model_version"),
        "app_version": row.get("app_version"),
        "evaluation_date": row.get("evaluation_date"),
        "configuration_id": row.get("configuration_id"),
    }


def _success(**payload: Any) -> Dict[str, Any]:
    return {"success": True, "interpretation_hints": INTERPRETATION_HINTS, **payload}


def _error(message: str, **extra: Any) -> Dict[str, Any]:
    return {"success": False, "error": message, "interpretation_hints": INTERPRETATION_HINTS, **extra}


async def list_evaluations(allowlist: List[int]) -> Dict[str, Any]:
    try:
        rows = await _aget("/v1/configuration/list", {"skip": 0, "limit": 200})
    except requests.RequestException as exc:
        return _error(f"HAIC API request failed: {exc}")

    if not isinstance(rows, list):
        return _error("Unexpected HAIC configuration list response.")

    evaluations = [
        _project_configuration(row)
        for row in rows
        if isinstance(row, dict) and row.get("id") in allowlist
    ]
    return _success(
        action="list_evaluations",
        pilot_configuration_ids=allowlist,
        evaluations=evaluations,
        count=len(evaluations),
    )


async def list_results(configuration_id: int, allowlist: List[int]) -> Dict[str, Any]:
    denied = assert_configuration_allowed(configuration_id, allowlist)
    if denied:
        return _error(denied)

    try:
        rows = await _aget(f"/v1/results/{configuration_id}")
    except requests.RequestException as exc:
        return _error(f"HAIC API request failed: {exc}")

    if not isinstance(rows, list):
        return _error("Unexpected HAIC results list response.")

    results = [_project_result_index(row) for row in rows if isinstance(row, dict)]
    if not results:
        return _success(
            action="list_results",
            configuration_id=configuration_id,
            results=[],
            message="No evaluation results stored yet for this configuration.",
        )
    return _success(action="list_results", configuration_id=configuration_id, results=results)


async def get_holistic(configuration_id: int, allowlist: List[int]) -> Dict[str, Any]:
    denied = assert_configuration_allowed(configuration_id, allowlist)
    if denied:
        return _error(denied)

    try:
        holistic = await _aget(f"/v1/results/{configuration_id}/holistic")
        result_rows = await _aget(f"/v1/results/{configuration_id}")
    except requests.RequestException as exc:
        return _error(f"HAIC API request failed: {exc}")

    if not isinstance(holistic, dict):
        return _error("Unexpected HAIC holistic response.")

    per_model: List[Dict[str, Any]] = []
    if isinstance(result_rows, list) and result_rows:
        tasks = [
            _aget(f"/v1/results/{configuration_id}/{row['id']}")
            for row in result_rows
            if isinstance(row, dict) and row.get("id") is not None
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        for row, raw in zip(result_rows, raw_results):
            if isinstance(raw, Exception) or not isinstance(raw, dict):
                continue
            per_model.append(
                {
                    "result_id": row.get("id"),
                    "ai_model_version": raw.get("ai_model_version"),
                    "app_versions": raw.get("app_versions"),
                    "evaluation_date": raw.get("generated_at"),
                    "haic": _extract_haic_metrics(raw),
                    "warnings": _dedupe_warnings(raw.get("warnings")),
                }
            )

    status_note = None
    if not per_model and holistic.get("haic"):
        status_note = "Holistic aggregate available; no per-model result breakdown found."

    return _success(
        action="get_holistic",
        configuration_id=configuration_id,
        evaluation_date=holistic.get("evaluation_date"),
        ai_model_version=holistic.get("ai_model_version"),
        haic=_extract_haic_metrics(holistic),
        warnings=_dedupe_warnings(holistic.get("warnings")),
        per_model_results=per_model,
        message=status_note,
    )


async def get_result(
    configuration_id: int, result_id: int, allowlist: List[int]
) -> Dict[str, Any]:
    denied = assert_configuration_allowed(configuration_id, allowlist)
    if denied:
        return _error(denied)

    try:
        raw = await _aget(f"/v1/results/{configuration_id}/{result_id}")
    except requests.RequestException as exc:
        return _error(f"HAIC API request failed: {exc}")

    if not isinstance(raw, dict):
        return _error("Unexpected HAIC result response.")

    if raw.get("configuration_id") not in (None, configuration_id):
        return _error("Result does not belong to the requested configuration.")

    return _success(
        action="get_result",
        configuration_id=configuration_id,
        result_id=result_id,
        ai_model_version=raw.get("ai_model_version"),
        app_versions=raw.get("app_versions"),
        evaluation_date=raw.get("generated_at"),
        source_log_path=raw.get("source_log_path"),
        haic=_extract_haic_metrics(raw),
        warnings=_dedupe_warnings(raw.get("warnings")),
        by_pillar=(raw.get("aggregates") or {}).get("by_pillar"),
    )


async def query_haic_benchmark(
    action: str,
    allowlist: List[int],
    configuration_id: Optional[int] = None,
    result_id: Optional[int] = None,
) -> Dict[str, Any]:
    if action == "list_evaluations":
        return await list_evaluations(allowlist)

    if not allowlist:
        return _error("No HAIC benchmark configurations are mapped to your account.")

    if configuration_id is None:
        if len(allowlist) == 1:
            configuration_id = allowlist[0]
        else:
            return _error(
                "configuration_id is required because your account maps to multiple HAIC evaluations. "
                "Call list_evaluations first."
            )

    if action == "list_results":
        return await list_results(configuration_id, allowlist)
    if action == "get_holistic":
        return await get_holistic(configuration_id, allowlist)
    if action == "get_result":
        if result_id is None:
            return _error("result_id is required for get_result.")
        return await get_result(configuration_id, result_id, allowlist)

    return _error(f"Unknown action: {action}")


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) < 10:
            return f"{value:.3f}"
        return f"{value:.2f}"
    return str(value)


def _format_haic_metrics_line(label: str, metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return f"- **{label}**: no metrics available"
    parts = []
    for key in ("Tr", "HCL", "F", "D", "EL", "A", "S", "EfficiencyScore"):
        if key in metrics:
            parts.append(f"{key}={_fmt_metric(metrics.get(key))}")
    return f"- **{label}**: " + ", ".join(parts)


def format_haic_result_markdown(result: Dict[str, Any]) -> str:
    """Turn a query_haic_benchmark payload into a user-facing markdown summary."""
    if not result.get("success"):
        lines = [
            "I could not retrieve your **HAIC Benchmark Suite** results.",
            f"**Reason:** {result.get('error', 'Unknown error')}",
        ]
        if result.get("user_email"):
            lines.append(f"**Logged-in as:** `{result['user_email']}`")
        return "\n\n".join(lines)

    action = result.get("action")
    lines = ["Here are your **HAIC Benchmark Suite** results:"]

    if action == "list_evaluations":
        evaluations = result.get("evaluations") or []
        if not evaluations:
            lines.append("\nNo HAIC evaluations are mapped to your account.")
            return "\n".join(lines)
        lines.append("")
        lines.append("| Application | Status | Pilot tag |")
        lines.append("| --- | --- | --- |")
        for row in evaluations:
            lines.append(
                f"| {row.get('application_name', '')} | {row.get('evaluation_status', '')} | {row.get('pilot_tag', '')} |"
            )
        return "\n".join(lines)

    if action == "list_results":
        rows = result.get("results") or []
        if not rows:
            lines.append(f"\n{result.get('message', 'No stored result runs yet.')}")
            return "\n".join(lines)
        lines.append("")
        lines.append("| Result ID | AI model version | App version |")
        lines.append("| --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row.get('id', '')} | {row.get('ai_model_version', '')} | {row.get('app_version', '')} |"
            )
        return "\n".join(lines)

    if action in ("get_holistic", "get_result"):
        lines.append("")
        lines.append(_format_haic_metrics_line("Overall", result.get("haic")))
        per_model = result.get("per_model_results") or []
        if per_model:
            lines.append("")
            lines.append("**Per AI model version:**")
            for row in per_model:
                label = row.get("ai_model_version") or f"result {row.get('result_id')}"
                lines.append(_format_haic_metrics_line(label, row.get("haic")))
        if result.get("message"):
            lines.append("")
            lines.append(str(result["message"]))
        return "\n".join(lines)

    return f"HAIC tool response:\n```json\n{json.dumps(result, indent=2)}\n```"
