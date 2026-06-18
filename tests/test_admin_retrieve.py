"""Route tests for pure retrieval (store faked, no embedder/Ollama)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiar.harness import admin_routes
from aiar.rag import store

AUTH = {"Authorization": "Bearer secret"}


class FakeRetrieveStore:
    StoreNotReady = store.StoreNotReady

    def __init__(self):
        self._inst = {"default": "published", "draft": "draft"}
        self.readable = True
        self.hits = []
        self.calls = []

    def descriptor(self, instance=None):
        status = self._inst.get(instance)
        if status is None:
            return None
        return SimpleNamespace(name=instance, display_name=instance, status=status)

    def ensure_readable(self):
        if not self.readable:
            raise store.StoreNotReady("embedder_unavailable", "embedder down")

    def query_scored(self, text, n_results=3, where=None, *, instance=None):
        self.calls.append({
            "text": text,
            "n_results": n_results,
            "where": where,
            "instance": instance,
        })
        return self.hits[:n_results]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    fake = FakeRetrieveStore()
    monkeypatch.setattr(admin_routes, "store", fake)
    app = FastAPI()
    app.include_router(admin_routes.router)
    c = TestClient(app)
    c._fake = fake
    return c


def _hit(**meta):
    return store.RetrievedChunk(
        id="abc-3",
        text="The divide function raises ValueError on division by zero.",
        score=0.81,
        metadata={
            "source": "README.md",
            "title": "Calculator",
            "index": 3,
            "category": "guide",
            "page_span": "[1, 2]",
            **meta,
        },
    )


def test_retrieve_requires_token(client):
    r = client.get("/instances/default/retrieve", params={"q": "divide"})
    assert r.status_code == 401


def test_retrieve_token_unset_returns_503(client, monkeypatch):
    monkeypatch.delenv("AIAR_SERVICE_TOKEN", raising=False)
    r = client.get("/instances/default/retrieve",
                   params={"q": "divide"}, headers=AUTH)
    assert r.status_code == 503


def test_retrieve_empty_query_400(client):
    r = client.get("/instances/default/retrieve", params={"q": "   "}, headers=AUTH)
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "empty_query"


def test_retrieve_post_empty_query_400(client):
    r = client.post("/instances/default/retrieve", json={"q": ""}, headers=AUTH)
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "empty_query"


def test_retrieve_unknown_instance_404(client):
    r = client.get("/instances/ghost/retrieve", params={"q": "divide"}, headers=AUTH)
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "unknown_instance"


def test_retrieve_draft_instance_404(client):
    r = client.get("/instances/draft/retrieve", params={"q": "divide"}, headers=AUTH)
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "unknown_instance"


def test_retrieve_503_when_store_or_embedder_not_ready(client):
    client._fake.readable = False
    r = client.get("/instances/default/retrieve",
                   params={"q": "divide"}, headers=AUTH)
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "embedder_unavailable"


def test_retrieve_published_empty_instance_returns_empty_success(client):
    r = client.get("/instances/default/retrieve",
                   params={"q": "divide"}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["count"] == 0
    assert r.json()["hits"] == []


def test_retrieve_serializes_ranked_chunks(client):
    client._fake.hits = [_hit()]
    r = client.get("/instances/default/retrieve",
                   params={"q": " divide ", "k": 4}, headers=AUTH)

    assert r.status_code == 200
    body = r.json()
    assert body["instance"] == "default"
    assert body["query"] == "divide"
    assert body["k"] == 4
    assert body["score_kind"] == "cosine_similarity"
    assert body["score_order"] == "desc"
    assert body["count"] == 1
    assert body["hits"] == [{
        "chunk_id": "abc-3",
        "source": "README.md",
        "title": "Calculator",
        "text": "The divide function raises ValueError on division by zero.",
        "score": 0.81,
        "chunk_index": 3,
        "category": "guide",
        "page_span": [1, 2],
        "metadata": {
            "source": "README.md",
            "title": "Calculator",
            "index": 3,
            "category": "guide",
            "page_span": "[1, 2]",
        },
    }]


def test_retrieve_clamps_k_and_passes_category_filter(client):
    client.get("/instances/default/retrieve",
               params={"q": "divide", "k": 999, "category": "guide"},
               headers=AUTH)
    assert client._fake.calls[-1] == {
        "text": "divide",
        "n_results": 50,
        "where": {"category": "guide"},
        "instance": "default",
    }


def test_retrieve_clamps_low_k_to_one(client):
    client.get("/instances/default/retrieve",
               params={"q": "divide", "k": 0}, headers=AUTH)
    assert client._fake.calls[-1]["n_results"] == 1


def test_retrieve_ignores_blank_category_filter(client):
    client.get("/instances/default/retrieve",
               params={"q": "divide", "category": "   "}, headers=AUTH)
    assert client._fake.calls[-1]["where"] is None


def test_retrieve_post_body_form(client):
    client._fake.hits = [_hit(page_span="")]
    r = client.post("/instances/default/retrieve",
                    json={"q": "divide", "k": 2, "category": "guide"},
                    headers=AUTH)
    assert r.status_code == 200
    assert r.json()["count"] == 1
    assert client._fake.calls[-1]["where"] == {"category": "guide"}
