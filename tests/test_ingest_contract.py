"""Contract tests for the ingest result shape (aiar.ingest.v1).

Covers the pure ``status_for`` semantics, the synchronous twin's status
distinctions (store/ingest faked), and that a polled job IS an IngestResult.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiar.contracts.ingest import (
    INGEST_SCHEMA_VERSION,
    UnknownInstance,
    serialize_ingest_result,
    status_for,
)
from aiar.harness import ingest_jobs
from aiar.rag import ingest_docs


def test_status_for_semantics():
    assert status_for(chunks_added=5, errors=[], publish_failed=False) == "done"
    # all-duplicate idempotent no-op is still success
    assert status_for(chunks_added=0, errors=[], publish_failed=False) == "done"
    # partial: added + errors -> done (caller must surface errors)
    assert status_for(chunks_added=3, errors=[{"e": 1}], publish_failed=False) == "done"
    # nothing landed + errors -> failed, never "ready"
    assert status_for(chunks_added=0, errors=[{"e": 1}], publish_failed=False) == "failed"
    assert status_for(chunks_added=9, errors=[], publish_failed=True) == "failed"
    assert status_for(chunks_added=0, errors=[], publish_failed=False, running=True) == "running"


def test_job_to_dict_is_ingest_result():
    job = ingest_jobs.new_job("proj", documents_total=2)
    job.chunks_added = 4
    job.duplicates = 1
    ingest_jobs.finish(job, "done")
    d = job.to_dict()
    assert d["schema_version"] == INGEST_SCHEMA_VERSION
    assert d["instance"] == "proj"
    assert d["accepted"] == 2
    assert d["chunks_added"] == 4
    assert d["duplicates"] == 1
    assert d["published"] is False
    assert d["status"] == "done"
    # transport-only keys preserved for back-compat
    assert d["job_id"] == job.job_id
    assert "started_at" in d and "ended_at" in d


class _FakeStore:
    class StoreNotReady(RuntimeError):
        pass

    def __init__(self):
        self._inst = {"proj": "draft"}
        self._seen = set()
        self.ingests = []

    def ensure_writable(self):
        return None

    def descriptor(self, instance=None):
        status = self._inst.get(instance)
        return None if status is None else SimpleNamespace(name=instance, status=status)

    def add(self, chunks, *, instance):
        key = (instance, chunks[0].metadata.get("document_hash"))
        if key in self._seen:
            return 0
        self._seen.add(key)
        return len(chunks)

    def publish_instance(self, name):
        self._inst[name] = "published"

    def record_ingest(self, name, *, error=None):
        self.ingests.append((name, error))


class _FakeIngest:
    def ingest_document(self, *, source, title, text, pages, category, metadata):
        if not (text or "").strip():
            return []
        return [SimpleNamespace(metadata={"document_hash": text})]


@pytest.fixture
def fakes(monkeypatch):
    fs, fi = _FakeStore(), _FakeIngest()
    monkeypatch.setattr(ingest_docs, "_store", fs)
    monkeypatch.setattr(ingest_docs, "_ingest", fi)
    return fs


def _doc(doc_id, text):
    return {"doc_id": doc_id, "source": "s", "title": "t", "text": text}


def test_twin_unknown_instance_raises(fakes):
    with pytest.raises(UnknownInstance):
        ingest_docs.ingest_documents([_doc("d", "hi")], instance="ghost")


def test_twin_added_done(fakes):
    out = ingest_docs.ingest_documents([_doc("d1", "alpha"), _doc("d2", "beta")],
                                       instance="proj")
    assert out["schema_version"] == INGEST_SCHEMA_VERSION
    assert out["status"] == "done"
    assert out["chunks_added"] == 2
    assert out["accepted"] == 2
    assert out["published"] is False  # publish defaults to False
    assert fakes.ingests == [("proj", None)]


def test_twin_all_empty_failed(fakes):
    out = ingest_docs.ingest_documents([_doc("e1", "  "), _doc("e2", "")],
                                       instance="proj")
    assert out["status"] == "failed"
    assert out["chunks_added"] == 0
    assert len(out["errors"]) == 2


def test_twin_partial_is_done_with_errors(fakes):
    out = ingest_docs.ingest_documents([_doc("good", "real"), _doc("bad", "")],
                                       instance="proj")
    assert out["status"] == "done"
    assert out["chunks_added"] == 1
    assert any(e["doc_id"] == "bad" for e in out["errors"])


def test_twin_publish_true_publishes_on_success(fakes):
    out = ingest_docs.ingest_documents([_doc("d", "real")], instance="proj",
                                       publish=True)
    assert out["published"] is True
    assert fakes._inst["proj"] == "published"


def test_twin_publish_skipped_when_nothing_landed(fakes):
    out = ingest_docs.ingest_documents([_doc("e", "")], instance="proj",
                                       publish=True)
    assert out["published"] is False
    assert out["status"] == "failed"
    assert fakes._inst["proj"] == "draft"
