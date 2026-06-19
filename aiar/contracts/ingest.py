"""``aiar.ingest.v1`` — ingest result wire contract.

One shape for the synchronous Python twin (``aiar.rag.ingest_documents``) and the
polled HTTP job (``GET /instances/{instance}/ingest-jobs/{job_id}``). A remote
caller reconstructs the synchronous result by polling the job to completion.
"""
from __future__ import annotations

from typing import List, Optional

INGEST_SCHEMA_VERSION = "aiar.ingest.v1"


class IngestError(Exception):
    """Base for ingest contract errors. ``code`` is the stable slug the HTTP
    layer maps to a status."""

    code = "ingest_error"
    http_status = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnknownInstance(IngestError):
    code = "unknown_instance"
    http_status = 404


def status_for(*, chunks_added: int, errors: list, publish_failed: bool,
               running: bool = False) -> str:
    """The single source of truth for ingest ``status`` (contract §3).

    - ``running``           -> "running"
    - publish was requested but failed             -> "failed"
    - 0 chunks added AND errors present            -> "failed" (never "ready")
    - otherwise (added, or all-duplicate idempotent no-op) -> "done"

    Note: "done" with a non-empty ``errors`` list is a *partial* success — the
    caller must surface the errors; status alone does not mean "all good".
    """
    if running:
        return "running"
    if publish_failed:
        return "failed"
    if chunks_added == 0 and errors:
        return "failed"
    return "done"


def serialize_ingest_result(*, instance: str, status: str, accepted: int,
                            chunks_added: int, duplicates: int,
                            errors: Optional[List[dict]] = None,
                            published: bool = False) -> dict:
    """Serialize an ``aiar.ingest.v1`` result envelope."""
    return {
        "schema_version": INGEST_SCHEMA_VERSION,
        "instance": instance,
        "status": status,
        "accepted": accepted,
        "chunks_added": chunks_added,
        "duplicates": duplicates,
        "errors": list(errors or []),
        "published": bool(published),
    }
