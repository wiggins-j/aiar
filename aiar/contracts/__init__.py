"""Wire contracts — one serializer per contract, shared by the in-process
Python API and the HTTP routes so in-process and remote callers get
byte-identical payloads.

Each module owns:
  - a ``*_SCHEMA_VERSION`` string (``aiar.<contract>.v1``),
  - the pure serializer(s) for that contract,
  - any typed errors carrying a stable ``code`` for the HTTP layer to map.

These modules import only stdlib + dataclasses — no FastAPI, no store — so they
stay usable from both the Python twins and the routes without a dependency cycle.
See ``docs/integration-contracts.md`` for the frozen shapes.
"""
from __future__ import annotations

from aiar.contracts.retrieve import (
    RETRIEVE_SCHEMA_VERSION,
    RetrieveError,
    EmptyQuery,
    UnknownInstance,
    serialize_hit,
    serialize_retrieve_result,
)
from aiar.contracts.grounding import (
    GROUNDING_SCHEMA_VERSION,
    GroundingRecord,
    serialize_grounding_record,
)
from aiar.contracts.ingest import (
    INGEST_SCHEMA_VERSION,
    serialize_ingest_result,
)
from aiar.contracts.capabilities import (
    CAPABILITIES_SCHEMA_VERSION,
    serialize_capabilities,
)

__all__ = [
    "RETRIEVE_SCHEMA_VERSION",
    "RetrieveError",
    "EmptyQuery",
    "UnknownInstance",
    "serialize_hit",
    "serialize_retrieve_result",
    "GROUNDING_SCHEMA_VERSION",
    "GroundingRecord",
    "serialize_grounding_record",
    "INGEST_SCHEMA_VERSION",
    "serialize_ingest_result",
    "CAPABILITIES_SCHEMA_VERSION",
    "serialize_capabilities",
]
