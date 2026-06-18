"""Unit tests for store.ensure_writable() and reserved-instance protection.

These mock the chromadb/embedder layer entirely so they run in a light dev venv
(no chromadb / sentence-transformers / torch) — matching the Mac-dev rule that
heavy runtime deps only load on the server.
"""
from __future__ import annotations

import pytest

from aiar.rag import store


@pytest.fixture
def writable_store(monkeypatch):
    """Pretend the store initialised (client + registry present) without loading
    chromadb. Embedder readiness is controlled per-test via _ensure_embedder."""
    monkeypatch.setattr(store, "_available", True)
    monkeypatch.setattr(store, "_client", object())
    monkeypatch.setattr(store, "_registry", object())
    yield


def test_ensure_writable_passes_when_embedder_loads(writable_store, monkeypatch):
    monkeypatch.setattr(store, "_ensure_embedder", lambda: True)
    store.ensure_writable()  # should not raise


def test_ensure_readable_passes_when_embedder_loads(writable_store, monkeypatch):
    monkeypatch.setattr(store, "_ensure_embedder", lambda: True)
    store.ensure_readable()  # should not raise


def test_ensure_readable_raises_when_embedder_fails(writable_store, monkeypatch):
    monkeypatch.setattr(store, "_ensure_embedder", lambda: False)
    with pytest.raises(store.StoreNotReady) as exc:
        store.ensure_readable()
    assert exc.value.code == "embedder_unavailable"


def test_ensure_readable_raises_when_store_unavailable(monkeypatch):
    monkeypatch.setattr(store, "_available", False)
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store, "_registry", None)
    monkeypatch.setattr(store, "init", lambda: None)
    with pytest.raises(store.StoreNotReady) as exc:
        store.ensure_readable()
    assert exc.value.code == "store_unavailable"


def test_ensure_writable_raises_when_embedder_fails(writable_store, monkeypatch):
    monkeypatch.setattr(store, "_ensure_embedder", lambda: False)
    with pytest.raises(store.StoreNotReady) as exc:
        store.ensure_writable()
    assert exc.value.code == "embedder_unavailable"


def test_ensure_writable_raises_when_store_unavailable(monkeypatch):
    monkeypatch.setattr(store, "_available", False)
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store, "_registry", None)
    # init() would try to import chromadb; stub it so the store stays unavailable
    # without doing real I/O.
    monkeypatch.setattr(store, "init", lambda: None)
    with pytest.raises(store.StoreNotReady) as exc:
        store.ensure_writable()
    assert exc.value.code == "store_unavailable"


def test_delete_reserved_instance_refused(writable_store, monkeypatch):
    # Canonicalise to the reserved slug without touching a real registry.
    monkeypatch.setattr(store, "RESERVED_INSTANCES", {"reserved"})
    monkeypatch.setattr(store, "_canonical_existing", lambda name: "reserved")
    with pytest.raises(ValueError, match="reserved"):
        store.delete_instance("reserved")


def test_delete_default_instance_refused(writable_store, monkeypatch):
    monkeypatch.setattr(store, "_canonical_existing", lambda name: "default")
    with pytest.raises(ValueError, match="default"):
        store.delete_instance("default")
