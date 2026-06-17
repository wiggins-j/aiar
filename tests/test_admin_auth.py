"""Unit tests for the bearer-token dependency (no chromadb/embedder)."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from aiar.harness import auth


def test_unset_token_disables_route_503(monkeypatch):
    monkeypatch.delenv("AIAR_SERVICE_TOKEN", raising=False)
    with pytest.raises(HTTPException) as exc:
        auth.require_token(authorization="Bearer anything")
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "ingest_disabled"


def test_empty_token_treated_as_unset_503(monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "")
    with pytest.raises(HTTPException) as exc:
        auth.require_token(authorization="Bearer x")
    assert exc.value.status_code == 503


def test_missing_header_401(monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    with pytest.raises(HTTPException) as exc:
        auth.require_token(authorization=None)
    assert exc.value.status_code == 401


def test_wrong_token_401(monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    with pytest.raises(HTTPException) as exc:
        auth.require_token(authorization="Bearer nope")
    assert exc.value.status_code == 401


def test_malformed_header_401(monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    with pytest.raises(HTTPException) as exc:
        auth.require_token(authorization="secret")  # no "Bearer " prefix
    assert exc.value.status_code == 401


def test_correct_token_passes(monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    assert auth.require_token(authorization="Bearer secret") is None


def test_bearer_prefix_case_insensitive(monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    assert auth.require_token(authorization="bearer secret") is None
