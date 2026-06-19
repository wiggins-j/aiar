"""Contract tests for the pure-retrieve twin + route parity (aiar.retrieve.v1).

No embedder / no Ollama: the store is faked. Asserts the Python twin and the HTTP
route return byte-identical payloads and that retrieval never calls generation.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiar.contracts.retrieve import (
    RETRIEVE_SCHEMA_VERSION,
    EmptyQuery,
    UnknownInstance,
)
from aiar.harness import admin_routes
from aiar.rag import pure_retrieve, store

AUTH = {"Authorization": "Bearer secret"}


class FakeStore:
    StoreNotReady = store.StoreNotReady

    def __init__(self):
        self._inst = {"default": "published", "draft": "draft"}

    def descriptor(self, instance=None):
        status = self._inst.get(instance)
        if status is None:
            return None
        return SimpleNamespace(name=instance, display_name=instance, status=status)

    def ensure_readable(self):
        return None

    def query_scored(self, text, n_results=3, where=None, *, instance=None):
        return [store.RetrievedChunk(
            id="c-0", text="hit text", score=0.77,
            metadata={"source": "a.md", "title": "A", "index": 0,
                      "category": "general", "page_span": "[1, 1]"})]


@pytest.fixture
def fake(monkeypatch):
    f = FakeStore()
    monkeypatch.setattr(pure_retrieve, "store", f)
    return f


def test_twin_carries_schema_version(fake):
    out = pure_retrieve.retrieve_chunks("query", instance="default")
    assert out["schema_version"] == RETRIEVE_SCHEMA_VERSION
    assert out["score_kind"] == "cosine_similarity"
    assert out["score_order"] == "desc"
    assert out["count"] == 1
    assert out["hits"][0]["page_span"] == [1, 1]


def test_twin_empty_query_raises(fake):
    with pytest.raises(EmptyQuery):
        pure_retrieve.retrieve_chunks("   ", instance="default")


def test_twin_unknown_and_draft_instance_raise(fake):
    with pytest.raises(UnknownInstance):
        pure_retrieve.retrieve_chunks("q", instance="ghost")
    with pytest.raises(UnknownInstance):
        pure_retrieve.retrieve_chunks("q", instance="draft")


def test_twin_and_route_are_byte_identical(fake, monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    monkeypatch.setattr(admin_routes, "store", fake)
    app = FastAPI()
    app.include_router(admin_routes.router)
    client = TestClient(app)

    twin = pure_retrieve.retrieve_chunks("query", instance="default", k=4)
    route = client.get("/instances/default/retrieve",
                       params={"q": "query", "k": 4}, headers=AUTH).json()
    assert twin == route


def test_retrieve_never_calls_generation(fake, monkeypatch):
    def boom(*a, **k):  # any generation call is a contract violation
        raise AssertionError("pure retrieve invoked a generation model")

    monkeypatch.setattr("aiar.llm.call_ollama", boom)
    out = pure_retrieve.retrieve_chunks("query", instance="default")
    assert out["count"] == 1
