"""In-memory ingest-job registry for the remote ingest API.

Decision #3: v1 ingest runs synchronously behind an async-shaped contract — the
POST runs inline, records a terminal :class:`JobRecord`, and returns
``202 {job_id}``; the job is already ``done``/``failed`` when the client polls.
Jobs are in-memory only (lost on restart), which is acceptable because ingest is
idempotent: re-posting the same documents adds 0 new chunks (``store.add`` dedups
by ``document_hash``). Phase-2 hook: persist to ``<knowledge>/ingest-jobs.jsonl``
for restart recovery.

Note on idempotency: there is intentionally **no separate doc_id manifest**.
``store.add`` already skips re-embedding a document whose ``document_hash`` and
chunk count are unchanged (``aiar/rag/store.py``), so a manifest would only
duplicate that — and would risk silently skipping a real re-ingest if it ever
diverged from the store (e.g. after delete + recreate). The chunk-layer
``document_hash`` dedup is the single source of truth.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from aiar.contracts.ingest import serialize_ingest_result


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    job_id: str
    instance: str
    status: str = "running"  # running | done | failed
    documents_total: int = 0
    chunks_added: int = 0
    duplicates: int = 0
    errors: List[dict] = field(default_factory=list)
    published: bool = False
    started_at: str = field(default_factory=_iso_now)
    ended_at: Optional[str] = None

    def to_dict(self) -> dict:
        """The polled job payload IS an ``aiar.ingest.v1`` result (contract §3),
        plus transport-only keys (job_id, timestamps). Additive over the
        IngestResult shape — existing keys are preserved for back-compat."""
        return {
            **serialize_ingest_result(
                instance=self.instance, status=self.status,
                accepted=self.documents_total, chunks_added=self.chunks_added,
                duplicates=self.duplicates, errors=self.errors,
                published=self.published),
            "job_id": self.job_id,
            "documents_total": self.documents_total,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


_JOBS: Dict[str, JobRecord] = {}
_JOBS_LOCK = threading.Lock()

# Cap the in-memory registry so a long-lived service can't accumulate job
# records forever. Insertion order is preserved, so eviction is FIFO (oldest
# first). v1 has one synchronous client; this is just a leak backstop.
_MAX_JOBS = 1000


def new_job(instance: str, *, documents_total: int = 0) -> JobRecord:
    job = JobRecord(job_id=uuid.uuid4().hex, instance=instance,
                    documents_total=documents_total)
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job
        while len(_JOBS) > _MAX_JOBS:
            _JOBS.pop(next(iter(_JOBS)))
    return job


def finish(job: JobRecord, status: str) -> None:
    # Field mutation here (and on job.chunks_added/errors in the route) is not
    # under _JOBS_LOCK. Safe under v1's contract: each request owns its job and
    # drives it to a terminal state before the client ever receives the job_id,
    # so no reader races a writer. Revisit when ingest moves to a background task
    # that the client polls concurrently (then guard field access too).
    job.status = status
    job.ended_at = _iso_now()


def get(instance: str, job_id: str) -> Optional[JobRecord]:
    """Return the job iff it exists AND belongs to ``instance`` (so a job_id from
    another corpus can't be read through the wrong instance path)."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None or job.instance != instance:
        return None
    return job


def reset_for_testing() -> None:
    with _JOBS_LOCK:
        _JOBS.clear()
