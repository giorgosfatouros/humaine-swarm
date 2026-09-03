"""Static mapping from pilot users (email / Keycloak group / MinIO policy) to HAIC configuration IDs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class HaicPilotEntry(TypedDict, total=False):
    pilot: str
    configuration_ids: List[int]
    emails: List[str]
    email_patterns: List[str]
    groups: List[str]
    policies: List[str]


# Match any of: exact emails, substrings in email, Keycloak group names, token policies.
# configuration_ids from live HAIC /api/v1/configuration/list (2026-09).
HAIC_PILOT_MAP: List[HaicPilotEntry] = [
    {
        "pilot": "smart_cities",
        "configuration_ids": [3],
        "emails": ["cities-pilot@humaine.com"],
        "email_patterns": ["smart-cities", "cities-pilot", "@cities"],
        "groups": ["smart-cities"],
        "policies": ["smart-cities"],
    },
    {
        "pilot": "smart_ticketing",
        "configuration_ids": [4],
        "emails": [],
        "email_patterns": ["smart-ticketing", "ticketing", "gft"],
        "groups": ["smart-coms-test"],
        "policies": ["smart-coms-test"],
    },
    {
        "pilot": "smart_healthcare_oncology",
        "configuration_ids": [7],
        "emails": [],
        "email_patterns": ["oncology", "healthentia"],
        "groups": [],
        "policies": [],
    },
    {
        "pilot": "smart_healthcare_diabetes",
        "configuration_ids": [2],
        "emails": [],
        "email_patterns": ["smart-healthcare", "healthcare-diabetes", "diabetes"],
        "groups": ["smart-healthcare"],
        "policies": ["smart-healthcare"],
    },
    {
        "pilot": "smart_energy",
        "configuration_ids": [6],
        "emails": [],
        "email_patterns": ["smart-energy", "energy"],
        "groups": ["smart-energy"],
        "policies": ["smart-energy"],
    },
    {
        "pilot": "smart_manufacturing",
        "configuration_ids": [5],
        "emails": [],
        "email_patterns": ["manufacturing", "smart-manufacturing"],
        "groups": ["smart-finance"],
        "policies": ["smart-finance"],
    },
    {
        "pilot": "radiology_demo",
        "configuration_ids": [1],
        "emails": [],
        "email_patterns": ["radiology"],
        "groups": ["services"],
        "policies": ["services"],
    },
    {
        "pilot": "new_pilot",
        "configuration_ids": [],
        "emails": [],
        "email_patterns": ["new-pilot"],
        "groups": ["new-pilot"],
        "policies": ["new-pilot-policy", "new-pilot"],
    },
]


def resolve_haic_pilot_context(
    user_email: str = "",
    groups: Optional[List[str]] = None,
    policies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Resolve HAIC access for the current user. First matching entry wins.

    Returns:
        pilot: matched pilot slug or None
        configuration_ids: allowlisted HAIC configuration IDs
        has_access: True when at least one configuration_id is mapped
        user_email: echoed for debugging
    """
    groups = groups or []
    policies = policies or []
    email_lower = (user_email or "").lower()

    for entry in HAIC_PILOT_MAP:
        config_ids = entry.get("configuration_ids") or []
        if not config_ids:
            continue

        for exact in entry.get("emails") or []:
            if email_lower and email_lower == exact.lower():
                return _context(entry, user_email)

        for pattern in entry.get("email_patterns") or []:
            if pattern.lower() in email_lower:
                return _context(entry, user_email)

        for group in groups:
            if group.lower() in {g.lower() for g in entry.get("groups") or []}:
                return _context(entry, user_email)

        policy_set = {p.lower() for p in policies}
        for policy in entry.get("policies") or []:
            if policy.lower() in policy_set:
                return _context(entry, user_email)

    return {
        "pilot": None,
        "configuration_ids": [],
        "has_access": False,
        "user_email": user_email,
    }


def _context(entry: HaicPilotEntry, user_email: str) -> Dict[str, Any]:
    return {
        "pilot": entry["pilot"],
        "configuration_ids": list(entry.get("configuration_ids") or []),
        "has_access": bool(entry.get("configuration_ids")),
        "user_email": user_email,
    }


def assert_configuration_allowed(configuration_id: int, allowlist: List[int]) -> Optional[str]:
    """Return an error message if configuration_id is not on the allowlist."""
    if configuration_id not in allowlist:
        return (
            f"Access denied: configuration_id {configuration_id} is not available for your pilot account."
        )
    return None
