"""``aiar.grounding.v1`` — grounding record wire contract.

The product-safe shape that keeps ``answer`` (what was wrong) and ``correction``
(the fix) as distinct fields, so a consumer can never accidentally store the
answer in the correction slot. Records are instance-scoped on disk by the store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

GROUNDING_SCHEMA_VERSION = "aiar.grounding.v1"


@dataclass
class GroundingRecord:
    """One grounding record. ``answer`` / ``prompt`` / ``source_chunks`` are
    optional and default to None/[] so legacy records (written before these
    fields existed) read back cleanly."""

    id: str
    signature: str
    normalized: str
    verdict: str                       # normalized rating: good | partial | bad
    correction: str = ""
    instance: Optional[str] = None
    reason: str = ""
    answer: Optional[str] = None
    prompt: Optional[str] = None
    source_chunks: List[str] = field(default_factory=list)
    failure_tags: List[str] = field(default_factory=list)
    confidence: str = "medium"
    created_at: str = ""

    def to_dict(self) -> dict:
        return serialize_grounding_record(self)


def serialize_grounding_record(rec: "GroundingRecord") -> dict:
    """Serialize a :class:`GroundingRecord` to the ``aiar.grounding.v1`` shape."""
    return {
        "schema_version": GROUNDING_SCHEMA_VERSION,
        "id": rec.id,
        "signature": rec.signature,
        "normalized": rec.normalized,
        "instance": rec.instance,
        "verdict": rec.verdict,
        "reason": rec.reason,
        "correction": rec.correction,
        "answer": rec.answer,
        "prompt": rec.prompt,
        "source_chunks": list(rec.source_chunks),
        "failure_tags": list(rec.failure_tags),
        "confidence": rec.confidence,
        "created_at": rec.created_at,
    }
