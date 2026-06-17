"""Unit tests for the in-memory ingest-job registry (no chromadb)."""
from __future__ import annotations

import pytest

from aiar.harness import ingest_jobs


@pytest.fixture(autouse=True)
def _clean_jobs():
    ingest_jobs.reset_for_testing()
    yield
    ingest_jobs.reset_for_testing()


def test_job_lifecycle():
    job = ingest_jobs.new_job("alpha", documents_total=3)
    assert job.status == "running"
    assert job.documents_total == 3
    assert ingest_jobs.get("alpha", job.job_id) is job
    job.chunks_added += 5
    ingest_jobs.finish(job, "done")
    fetched = ingest_jobs.get("alpha", job.job_id)
    assert fetched.status == "done"
    assert fetched.ended_at is not None
    assert fetched.chunks_added == 5


def test_get_wrong_instance_returns_none():
    job = ingest_jobs.new_job("alpha")
    assert ingest_jobs.get("beta", job.job_id) is None


def test_get_unknown_job_returns_none():
    assert ingest_jobs.get("alpha", "deadbeef") is None


def test_unique_job_ids():
    a = ingest_jobs.new_job("x")
    b = ingest_jobs.new_job("x")
    assert a.job_id != b.job_id
