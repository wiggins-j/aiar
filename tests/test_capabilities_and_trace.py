"""Tests for the capability manifest, /healthz markers, and /calls trace."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aiar.contracts.capabilities import CAPABILITIES_SCHEMA_VERSION
from aiar.contracts.retrieve import RETRIEVE_SCHEMA_VERSION
from aiar.harness import service


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(service, "healthcheck", lambda: True)
    return TestClient(service.app)


def test_manifest_reflects_token_toggle(client, monkeypatch):
    # With a token configured, the authenticated capabilities flip on.
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    m = client.get("/capabilities").json()
    assert m["schema_version"] == CAPABILITIES_SCHEMA_VERSION
    assert m["features"]["pure_retrieve"] is True
    assert m["features"]["remote_ingest"] is True
    # Deferred (A5) features must report false so consumers don't build on them.
    assert m["features"]["semantic_grounding"] is False
    assert m["features"]["streaming"] is False
    assert m["schemas"]["retrieve"] == RETRIEVE_SCHEMA_VERSION
    assert m["backend_id"]

    # Unset the token: the manifest must stop claiming usable auth'd features.
    monkeypatch.delenv("AIAR_SERVICE_TOKEN", raising=False)
    m2 = client.get("/capabilities").json()
    assert m2["features"]["pure_retrieve"] is False
    assert m2["features"]["remote_ingest"] is False


def test_healthz_exposes_retrieve_schema_version(client, monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    h = client.get("/healthz").json()
    assert h["retrieve_schema_version"] == RETRIEVE_SCHEMA_VERSION
    assert "pure_retrieve" in h and "remote_ingest" in h


def test_calls_unknown_id_404(client, monkeypatch):
    monkeypatch.setattr(service.observer, "read_by_call_id", lambda cid: None)
    r = client.get("/calls/nope")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "unknown_call"


def test_calls_redacts_bytes_by_default(client, monkeypatch):
    event = {
        "call_id": "cid", "timestamp": "t", "endpoint": "/services/prompt",
        "model": "m", "prompt_tokens": 10, "completion_tokens": 20,
        "latency_ms": 5, "raw_prompt": "SECRET PROMPT",
        "system_prompt": "SYS", "user_prompt": "USR", "response_text": "ANS",
        "thinking": "THINK", "error": None,
    }
    monkeypatch.setattr(service.observer, "read_by_call_id", lambda cid: event)
    monkeypatch.delenv("AIAR_TRACE_DEBUG", raising=False)
    t = client.get("/calls/cid").json()
    assert t["prompt_tokens"] == 10 and t["completion_tokens"] == 20
    assert t["debug"] is False
    for key in ("raw_prompt", "system_prompt", "user_prompt", "response_text",
                "thinking"):
        assert "redacted" in t[key]
        assert "SECRET" not in t[key]


def test_calls_reveals_bytes_in_debug(client, monkeypatch):
    event = {"call_id": "cid", "raw_prompt": "SECRET PROMPT", "error": None}
    monkeypatch.setattr(service.observer, "read_by_call_id", lambda cid: event)
    monkeypatch.setenv("AIAR_TRACE_DEBUG", "1")
    t = client.get("/calls/cid").json()
    assert t["debug"] is True
    assert t["raw_prompt"] == "SECRET PROMPT"
