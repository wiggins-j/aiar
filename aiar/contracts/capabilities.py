"""``aiar.capabilities.v1`` — capability manifest wire contract.

A consuming app gates UI affordances on this, never on a version string.
``features`` are derived from live predicates (mounted **and** usable), so the
manifest can never claim a capability the process can't actually serve.
"""
from __future__ import annotations

from typing import Optional

from aiar.contracts.retrieve import RETRIEVE_SCHEMA_VERSION
from aiar.contracts.ingest import INGEST_SCHEMA_VERSION
from aiar.contracts.grounding import GROUNDING_SCHEMA_VERSION

CAPABILITIES_SCHEMA_VERSION = "aiar.capabilities.v1"
ANSWER_SCHEMA_VERSION = "aiar.answer.v1"


def serialize_capabilities(*, aiar_version: str, backend_id: str,
                           pure_retrieve: bool, remote_ingest: bool,
                           grounding_v1: bool = True,
                           semantic_grounding: bool = False,
                           judge_only: bool = False, streaming: bool = False,
                           answer_sources: bool = True,
                           call_trace: bool = True,
                           generation: bool = True) -> dict:
    """Serialize the ``aiar.capabilities.v1`` manifest."""
    return {
        "schema_version": CAPABILITIES_SCHEMA_VERSION,
        "aiar_version": aiar_version,
        "backend_id": backend_id,
        "features": {
            "pure_retrieve": bool(pure_retrieve),
            "remote_ingest": bool(remote_ingest),
            "grounding_v1": bool(grounding_v1),
            "semantic_grounding": bool(semantic_grounding),  # A5, deferred
            "judge_only": bool(judge_only),                  # A5
            "streaming": bool(streaming),                    # A5
            "answer_sources": bool(answer_sources),
            "call_trace": bool(call_trace),
            # Generation readiness: is the active model actually pulled?
            "generation": bool(generation),
        },
        "schemas": {
            "retrieve": RETRIEVE_SCHEMA_VERSION,
            "ingest": INGEST_SCHEMA_VERSION,
            "grounding": GROUNDING_SCHEMA_VERSION,
            "answer": ANSWER_SCHEMA_VERSION,
        },
    }
