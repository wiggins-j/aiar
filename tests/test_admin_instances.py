"""Route tests for instance management + auth gating, via TestClient.

The store layer is faked in-memory (no chromadb / embedder) so these run in the
light Mac dev venv. Embedding/ChromaDB integration is exercised on the server.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aiar.harness import admin_routes, ingest_jobs
from aiar.rag import store


class FakeStore:
    """Minimal in-memory stand-in for the slice of aiar.rag.store the admin
    routes call. Mirrors the contract, not the implementation."""

    StoreNotReady = store.StoreNotReady

    def __init__(self):
        self._inst = {"default": "published", "reserved": "published"}
        self.added = []          # (instance, n_chunks)
        self._seen_hashes = set()  # emulate store.add's document_hash dedup
        self.writable = True
        self.ingests = []        # (instance, error) — last-ingest telemetry

    # lifecycle
    def ensure_writable(self):
        if not self.writable:
            raise store.StoreNotReady("embedder_unavailable", "embedder down")

    def create_instance(self, name, *, display_name=None, query_rewrite=None,
                        rerank_model=None):
        from aiar.rag import instances
        slug = instances.slugify(name)
        self._inst.setdefault(slug, "draft")
        return slug

    def publish_instance(self, name):
        self._inst[name] = "published"

    def record_ingest(self, name, *, error=None):
        self.ingests.append((name, error))

    def delete_instance(self, name):
        if name == "default":
            raise ValueError("cannot delete the default instance")
        if name == "reserved":
            raise ValueError("cannot delete reserved instance: 'reserved'")
        self._inst.pop(name, None)
        return {"deleted": name, "active": "default"}

    def descriptor(self, instance=None):
        status = self._inst.get(instance)
        if status is None:
            return None
        return SimpleNamespace(name=instance, display_name=instance, status=status)

    def list_instances(self):
        return [{"name": n, "display_name": n, "status": s,
                 "chunk_count": 0, "active": n == "default"}
                for n, s in self._inst.items()]

    def health(self, *, instance=None):
        return {"store_ready": True, "embedder_ready": True,
                "embedding_model": "all-MiniLM-L6-v2", "chunk_count": 0,
                "active_instance": "default", "resolved_instance": instance}

    def add(self, chunks, *, instance):
        # Mirror store.add: a doc whose document_hash is already present adds 0
        # (all chunks of one document share the hash).
        if not chunks:
            return 0
        key = (instance, chunks[0].metadata.get("document_hash"))
        if key in self._seen_hashes:
            return 0
        self._seen_hashes.add(key)
        self.added.append((instance, len(chunks)))
        return len(chunks)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    fake = FakeStore()
    monkeypatch.setattr(admin_routes, "store", fake)
    ingest_jobs.reset_for_testing()
    # build the app fresh so startup hooks don't touch real chromadb
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(admin_routes.router)
    c = TestClient(app)
    c._fake = fake
    return c


AUTH = {"Authorization": "Bearer secret"}


def test_requires_token(client):
    assert client.get("/instances").status_code == 401
    assert client.get("/instances", headers=AUTH).status_code == 200


def test_token_unset_returns_503(client, monkeypatch):
    monkeypatch.delenv("AIAR_SERVICE_TOKEN", raising=False)
    assert client.get("/instances", headers=AUTH).status_code == 503


def test_create_idempotent_created_flag(client):
    r1 = client.post("/instances", json={"name": "Remote Proj"}, headers=AUTH)
    assert r1.status_code == 200
    assert r1.json() == {"instance": "remote-proj", "status": "draft", "created": True}
    r2 = client.post("/instances", json={"name": "Remote Proj"}, headers=AUTH)
    assert r2.json()["created"] is False


def test_list_includes_published_flag(client):
    out = client.get("/instances", headers=AUTH).json()["instances"]
    by_name = {d["name"]: d for d in out}
    assert by_name["default"]["published"] is True


def test_health_unknown_instance_404(client):
    assert client.get("/instances/nope/health", headers=AUTH).status_code == 404


def test_health_known_instance(client):
    r = client.get("/instances/default/health", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["published"] is True


def test_publish_unknown_404(client):
    assert client.post("/instances/nope/publish", headers=AUTH).status_code == 404


def test_publish_known(client):
    client.post("/instances", json={"name": "p"}, headers=AUTH)
    r = client.post("/instances/p/publish", headers=AUTH)
    assert r.json() == {"instance": "p", "published": True}


def test_delete_default_and_reserved_protected_400(client):
    assert client.delete("/instances/default", headers=AUTH).status_code == 400
    assert client.delete("/instances/reserved", headers=AUTH).status_code == 400


def test_delete_unknown_404(client):
    assert client.delete("/instances/ghost", headers=AUTH).status_code == 404


def test_delete_ok(client):
    client.post("/instances", json={"name": "tmp"}, headers=AUTH)
    r = client.delete("/instances/tmp", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["deleted"] is True
