"""Tests for logging users out once their Keycloak token has expired."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from fastapi import FastAPI, HTTPException

from chainlit.auth import get_current_user
from classes.user_handler import UserSessionManager
from utils.session_guard import (
    get_current_user_with_live_token,
    install_session_guard,
    is_oauth_token_expired,
)


def _token(exp_offset: float) -> str:
    return jwt.encode({"sub": "alice", "exp": int(time.time() + exp_offset)}, "k" * 32, algorithm="HS256")


def _user(oauth_token=None):
    metadata = {"oauth_token": oauth_token} if oauth_token else {}
    return SimpleNamespace(identifier="alice", metadata=metadata)


def _guard(user):
    with patch("utils.session_guard.get_current_user", new=AsyncMock(return_value=user)) as inner:
        result = asyncio.run(get_current_user_with_live_token("chainlit-jwt"))
    inner.assert_awaited_once_with("chainlit-jwt")
    return result


def test_token_expiry_detection():
    assert is_oauth_token_expired(_token(-60))
    assert not is_oauth_token_expired(_token(600))
    assert not is_oauth_token_expired(jwt.encode({"sub": "alice"}, "k" * 32, algorithm="HS256"))
    assert not is_oauth_token_expired("not-a-jwt")
    assert not is_oauth_token_expired(None)


def test_live_token_passes_through():
    user = _user(_token(600))
    assert _guard(user) is user


@pytest.mark.parametrize("user", [_user(), None])
def test_users_without_keycloak_token_pass_through(user):
    assert _guard(user) is user


def test_expired_token_answers_401():
    with pytest.raises(HTTPException) as raised:
        _guard(_user(_token(-60)))
    assert raised.value.status_code == 401


def test_guard_overrides_chainlit_dependency():
    app = FastAPI()
    install_session_guard(app)
    assert app.dependency_overrides[get_current_user] is get_current_user_with_live_token


def test_clear_session_drops_token_and_derived_state():
    stored = {}
    with patch("classes.user_handler.cl.user_session.set", side_effect=stored.__setitem__):
        UserSessionManager.clear_session()
    assert stored["oauth_token"] is None
    assert stored["minio_credentials"] is None
    assert stored["kubeflow_namespaces"] is None
    assert stored["user_policies"] == []
