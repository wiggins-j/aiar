"""Regression: mounting the admin router must not change the existing
query/eval/health routes (AC#6). Heavy deps mocked; runs in the light venv."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aiar.harness import service
from aiar.rag import store


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(service, "answer_prompt",
                        lambda prompt, **kw: {"answer": "ok", "grounded": kw.get("rag", True)})
    monkeypatch.setattr(service, "healthcheck", lambda: True)
    monkeypatch.setattr(service.store, "health", lambda: {"store_ready": True})
    monkeypatch.setattr(service.store, "init", lambda: None)  # no chromadb on startup
    return TestClient(service.app)


def test_services_prompt_still_works(client):
    r = client.post("/services/prompt", json={"service_name": "X", "prompt": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "ok"
    assert body["service_name"] == "X"


def test_eval_prompt_still_works(client):
    r = client.post("/eval/prompt", json={"prompt": "hi"})
    assert r.status_code == 200
    assert r.json()["answer"] == "ok"


def test_healthz_unchanged(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["rag"]["store_ready"] is True


def test_healthz_remote_ingest_true_when_mounted_and_token_set(client, monkeypatch):
    # routes mounted AND token configured -> ingest is actually usable (§9.8)
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    assert client.get("/healthz").json()["remote_ingest"] is True
    assert client.get("/healthz").json()["pure_retrieve"] is True


def test_healthz_remote_ingest_false_when_token_unset(client, monkeypatch):
    # routes mounted but NO token -> every write 503s, so must report false
    monkeypatch.delenv("AIAR_SERVICE_TOKEN", raising=False)
    assert client.get("/healthz").json()["remote_ingest"] is False
    assert client.get("/healthz").json()["pure_retrieve"] is False


def test_remote_ingest_mounted_false_without_routes(monkeypatch):
    # a query-only app (router NOT mounted) is not ingest-capable regardless
    from fastapi import FastAPI
    bare = FastAPI()
    monkeypatch.setattr(service, "app", bare)
    assert service._remote_ingest_mounted() is False
    assert service._pure_retrieve_mounted() is False


def test_admin_routes_are_mounted(client):
    # present but auth-gated (503 because no token set, not 404)
    r = client.get("/instances")
    assert r.status_code in (401, 503)


def test_retrieve_route_does_not_call_generation(monkeypatch):
    class FakeStore:
        def health(self):
            return {"store_ready": True}

        def init(self):
            return None

        def descriptor(self, instance=None):
            if instance == "default":
                return SimpleNamespace(name="default", status="published")
            return None

        def ensure_readable(self):
            return None

        def query_scored(self, text, n_results=3, where=None, *, instance=None):
            return [store.RetrievedChunk(
                id="id-0",
                text="retrieved chunk",
                score=0.9,
                metadata={"source": "doc.md", "title": "Doc", "index": 0,
                          "category": "general"},
            )]

    def fail_generation(*args, **kwargs):
        raise AssertionError("pure retrieve must not call answer_prompt")

    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    monkeypatch.setattr(service, "answer_prompt", fail_generation)
    monkeypatch.setattr(service, "healthcheck", lambda: True)
    monkeypatch.setattr(service, "store", FakeStore())
    from aiar.harness import admin_routes
    monkeypatch.setattr(admin_routes, "store", service.store)

    c = TestClient(service.app)
    r = c.get("/instances/default/retrieve",
              params={"q": "anything"}, headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
    assert r.json()["hits"][0]["text"] == "retrieved chunk"
