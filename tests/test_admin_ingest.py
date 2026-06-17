"""Route tests for document ingest + job status (store faked, no embedder)."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiar.harness import admin_routes, ingest_jobs
from tests.test_admin_instances import FakeStore

AUTH = {"Authorization": "Bearer secret"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    fake = FakeStore()
    fake._inst["proj"] = "draft"
    monkeypatch.setattr(admin_routes, "store", fake)
    ingest_jobs.reset_for_testing()
    app = FastAPI()
    app.include_router(admin_routes.router)
    c = TestClient(app)
    c._fake = fake
    return c


def _docs(*texts):
    return {"documents": [
        {"doc_id": f"d{i}", "source": f"s{i}", "title": f"t{i}", "text": t}
        for i, t in enumerate(texts)]}


def test_ingest_unknown_instance_404(client):
    r = client.post("/instances/ghost/documents", json=_docs("hi"), headers=AUTH)
    assert r.status_code == 404


def test_ingest_requires_token(client):
    assert client.post("/instances/proj/documents", json=_docs("hi")).status_code == 401


def test_ingest_503_when_not_writable(client):
    client._fake.writable = False
    r = client.post("/instances/proj/documents", json=_docs("hi"), headers=AUTH)
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "embedder_unavailable"


def test_ingest_happy_path_job_done(client):
    r = client.post("/instances/proj/documents",
                    json=_docs("hello world", "another doc"), headers=AUTH)
    assert r.status_code == 202
    body = r.json()
    assert body["accepted"] == 2 and body["instance"] == "proj"
    job = client.get(f"/instances/proj/ingest-jobs/{body['job_id']}", headers=AUTH).json()
    assert job["status"] == "done"
    assert job["chunks_added"] >= 2
    assert job["duplicates"] == 0


def test_ingest_idempotent_redelivery_adds_zero(client):
    payload = _docs("the quick brown fox")
    first = client.post("/instances/proj/documents", json=payload, headers=AUTH).json()
    j1 = client.get(f"/instances/proj/ingest-jobs/{first['job_id']}", headers=AUTH).json()
    assert j1["chunks_added"] >= 1
    second = client.post("/instances/proj/documents", json=payload, headers=AUTH).json()
    j2 = client.get(f"/instances/proj/ingest-jobs/{second['job_id']}", headers=AUTH).json()
    assert j2["chunks_added"] == 0
    assert j2["duplicates"] >= 1


def test_ingest_empty_doc_recorded_in_errors_not_aborting(client):
    payload = {"documents": [
        {"doc_id": "good", "source": "s", "title": "t", "text": "real content"},
        {"doc_id": "empty", "source": "s2", "title": "t2", "text": "   "}]}
    r = client.post("/instances/proj/documents", json=payload, headers=AUTH).json()
    job = client.get(f"/instances/proj/ingest-jobs/{r['job_id']}", headers=AUTH).json()
    assert job["chunks_added"] >= 1               # good doc still ingested
    assert any(e["doc_id"] == "empty" for e in job["errors"])


def test_ingest_all_empty_docs_job_failed(client):
    # nothing stored + errors -> must NOT report success (AC#4)
    payload = {"documents": [
        {"doc_id": "e1", "source": "s", "text": "   "},
        {"doc_id": "e2", "source": "s2", "text": ""}]}
    r = client.post("/instances/proj/documents", json=payload, headers=AUTH).json()
    job = client.get(f"/instances/proj/ingest-jobs/{r['job_id']}", headers=AUTH).json()
    assert job["chunks_added"] == 0
    assert job["status"] == "failed"
    assert len(job["errors"]) == 2


def test_publish_skipped_when_batch_wholly_failed(client):
    # publish=True must NOT publish a draft when nothing landed and docs errored
    payload = {"documents": [{"doc_id": "e1", "source": "s", "text": ""}],
               "publish": True}
    client.post("/instances/proj/documents", json=payload, headers=AUTH)
    assert client._fake._inst["proj"] == "draft"  # still a draft, not published


def test_ingest_too_many_documents_413(client):
    payload = {"documents": [
        {"doc_id": f"d{i}", "source": f"s{i}", "text": "x"} for i in range(201)]}
    r = client.post("/instances/proj/documents", json=payload, headers=AUTH)
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "too_many_documents"


def test_ingest_publish_flag(client):
    r = client.post("/instances/proj/documents",
                    json={**_docs("content"), "publish": True}, headers=AUTH).json()
    assert client._fake._inst["proj"] == "published"
    job = client.get(f"/instances/proj/ingest-jobs/{r['job_id']}", headers=AUTH).json()
    assert job["status"] == "done"


def test_unknown_job_404(client):
    assert client.get("/instances/proj/ingest-jobs/deadbeef",
                      headers=AUTH).status_code == 404


def test_pages_document_ingests(client):
    payload = {"documents": [{
        "doc_id": "pdf1", "source": "s", "title": "t",
        "pages": [{"page": 1, "text": "page one text"},
                  {"page": 2, "text": "page two text"}]}]}
    r = client.post("/instances/proj/documents", json=payload, headers=AUTH).json()
    job = client.get(f"/instances/proj/ingest-jobs/{r['job_id']}", headers=AUTH).json()
    assert job["status"] == "done"
    assert job["chunks_added"] >= 1
