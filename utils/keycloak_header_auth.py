"""
Validate Keycloak OIDC access tokens for Chainlit header authentication (embed flow).
Uses realm JWKS from the token issuer (iss claim).

JWKS is fetched over HTTPS. If you see SSL: CERTIFICATE_VERIFY_FAILED when Chainlit
loads signing keys:

  - Preferred: install your org CA or set KEYCLOAK_CA_BUNDLE=/path/to/ca.pem
  - Local dev only: KEYCLOAK_JWKS_SSL_VERIFY=false (disables TLS verification for JWKS)
"""

from __future__ import annotations

import logging
import os
import ssl
from typing import Any, Dict, Optional

import jwt
from jwt import PyJWKClient

logger = logging.getLogger(__name__)


def _jwks_ssl_context() -> Optional[ssl.SSLContext]:
    """
    SSL context for PyJWKClient when fetching .../openid-connect/certs.
    Returns None to use urllib/PyJWT defaults (system trust store).
    """
    bundle = os.environ.get("KEYCLOAK_CA_BUNDLE") or os.environ.get(
        "REQUESTS_CA_BUNDLE"
    )
    verify_raw = os.environ.get("KEYCLOAK_JWKS_SSL_VERIFY", "true").lower()
    insecure = verify_raw in ("0", "false", "no", "off")

    if bundle and os.path.isfile(bundle):
        logger.info("Keycloak JWKS: using CA bundle %s", bundle)
        return ssl.create_default_context(cafile=bundle)

    if insecure:
        logger.warning(
            "Keycloak JWKS: KEYCLOAK_JWKS_SSL_VERIFY=false — TLS verification disabled "
            "for JWKS fetch only (use only on trusted networks / local dev)"
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    return None


def decode_keycloak_access_token(access_token: str) -> Dict[str, Any]:
    """
    Verify RS256 signature using the issuer's JWKS and return claims.
    Audience is not enforced (Keycloak clients vary); issuer and expiry are verified.
    """
    unverified = jwt.decode(
        access_token,
        options={"verify_signature": False},
    )
    iss = unverified.get("iss")
    if not iss or not isinstance(iss, str):
        raise ValueError("Token missing iss claim")

    jwks_url = f"{iss.rstrip('/')}/protocol/openid-connect/certs"
    ssl_ctx = _jwks_ssl_context()
    jwks_client = (
        PyJWKClient(jwks_url, ssl_context=ssl_ctx)
        if ssl_ctx is not None
        else PyJWKClient(jwks_url)
    )
    signing_key = jwks_client.get_signing_key_from_jwt(access_token)

    decoded = jwt.decode(
        access_token,
        signing_key.key,
        algorithms=["RS256"],
        issuer=iss,
        options={"verify_aud": False},
    )
    return decoded


def claims_to_display_name(claims: Dict[str, Any]) -> Optional[str]:
    return (
        claims.get("name")
        or claims.get("preferred_username")
        or claims.get("email")
    )


def claims_to_identifier(claims: Dict[str, Any]) -> str:
    return (
        claims.get("email")
        or claims.get("preferred_username")
        or claims.get("sub")
        or "unknown"
    )


def validate_bearer_token(access_token: str) -> Optional[Dict[str, Any]]:
    try:
        return decode_keycloak_access_token(access_token)
    except Exception as e:
        # Use ERROR so operators see failures when CHAT logger is set to ERROR (default in app.py)
        logger.error("Keycloak token validation failed: %s", e, exc_info=True)
        return None
