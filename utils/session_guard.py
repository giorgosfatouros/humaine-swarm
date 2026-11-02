"""
Log users out of Chainlit once the Keycloak token in their session has expired.

Chainlit's own login cookie outlives the Keycloak access token stored in the user
metadata, so without this a refresh restores a session whose token MinIO and
Kubeflow reject. The guard overrides Chainlit's ``get_current_user`` dependency:
HTTP calls (``/user`` on page load included) answer 401, and the UI shows the
login page again.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import jwt
from fastapi import Depends, FastAPI, HTTPException

from chainlit.auth import get_current_user, reuseable_oauth

logger = logging.getLogger(__name__)


def is_oauth_token_expired(token: Optional[str]) -> bool:
    """True when the token is a JWT whose ``exp`` has passed. Undecodable tokens count as live."""
    if not token:
        return False
    try:
        claims = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError:
        return False
    exp = claims.get("exp")
    return isinstance(exp, (int, float)) and exp <= time.time()


def _oauth_token_of(user: Any) -> Optional[str]:
    metadata = getattr(user, "metadata", None)
    return metadata.get("oauth_token") if isinstance(metadata, dict) else None


async def get_current_user_with_live_token(token: Optional[str] = Depends(reuseable_oauth)):
    """Chainlit's ``get_current_user``, refusing users whose Keycloak token has expired."""
    user = await get_current_user(token)
    if is_oauth_token_expired(_oauth_token_of(user)):
        logger.info("Keycloak token expired for %s; asking the UI to log in again", user.identifier)
        raise HTTPException(status_code=401, detail="Keycloak session expired")
    return user


def install_session_guard(app: FastAPI) -> None:
    """Route every Chainlit endpoint that needs the current user through the expiry check."""
    app.dependency_overrides[get_current_user] = get_current_user_with_live_token
