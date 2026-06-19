"""``aiar.retrieve.v1`` — pure-retrieval wire contract.

One serializer for the hit shape + the result envelope, shared by the HTTP route
(``GET/POST /instances/{instance}/retrieve``) and the in-process twin
(``aiar.rag.retrieve_chunks``). Pure retrieve is **raw vector similarity**: it
never invokes a generation model and does not run the answerer's hybrid/rerank
pipeline (see the contract doc, §1).
"""
from __future__ import annotations

import json
from typing import Any, List, Optional

RETRIEVE_SCHEMA_VERSION = "aiar.retrieve.v1"

# Score semantics are constant for this route: raw cosine similarity, higher is
# better. They ride the response (not each hit) and are advertised via the
# capability manifest's ``schemas.retrieve`` — deliberately NOT a global /healthz
# constant, since a future ranking change would invalidate a standalone constant.
SCORE_KIND = "cosine_similarity"
SCORE_ORDER = "desc"


class RetrieveError(Exception):
    """Base for retrieval contract errors. ``code`` is the stable slug the HTTP
    layer maps to a status; ``http_status`` is the matching code."""

    code = "retrieve_error"
    http_status = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class EmptyQuery(RetrieveError):
    code = "empty_query"
    http_status = 400


class UnknownInstance(RetrieveError):
    code = "unknown_instance"
    http_status = 404


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_page_span(value: Any) -> Optional[List[int]]:
    """Coerce a stored page_span (list, tuple, or JSON string) to ``[start, end]``
    or None. Tolerant: anything that isn't a 2-element int pair becomes None."""
    if value is None or value == "":
        return None
    raw = value
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except ValueError:
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        return [int(raw[0]), int(raw[1])]
    except (TypeError, ValueError):
        return None


def serialize_hit(hit: Any) -> dict:
    """Serialize one ``store.RetrievedChunk`` into the wire hit shape.

    Duck-typed against ``RetrievedChunk`` (``id`` / ``text`` / ``score`` /
    ``metadata``) so this module need not import the store.
    """
    meta = dict(getattr(hit, "metadata", None) or {})
    return {
        "chunk_id": getattr(hit, "id", ""),
        "source": str(meta.get("source") or ""),
        "title": str(meta.get("title") or ""),
        "text": getattr(hit, "text", ""),
        "score": getattr(hit, "score", 0.0),
        "chunk_index": _optional_int(meta.get("index")),
        "category": str(meta.get("category") or "general"),
        "page_span": _parse_page_span(meta.get("page_span")),
        "metadata": meta,
    }


def serialize_retrieve_result(*, instance: str, query: str, k: int,
                              hits: List[Any]) -> dict:
    """Serialize a full ``aiar.retrieve.v1`` result envelope."""
    out = [serialize_hit(h) for h in hits]
    return {
        "schema_version": RETRIEVE_SCHEMA_VERSION,
        "instance": instance,
        "query": query,
        "k": k,
        "count": len(out),
        "score_kind": SCORE_KIND,
        "score_order": SCORE_ORDER,
        "hits": out,
    }
