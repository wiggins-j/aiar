"""Pure retrieval — the in-process twin of ``GET/POST /instances/{i}/retrieve``.

Raw vector similarity over a published instance's collection. **Never invokes a
generation model** and does not run the answerer's hybrid/rerank/rewrite pipeline
(see ``docs/integration-contracts.md`` §1) — it calls ``store.query_scored``
directly. Shares the ``aiar.retrieve.v1`` serializer with the HTTP route, so the
in-process and remote payloads are byte-identical.
"""
from __future__ import annotations

from typing import Optional

from aiar.contracts.retrieve import (
    EmptyQuery,
    UnknownInstance,
    serialize_retrieve_result,
)
from aiar.rag import instances, store

_DEFAULT_K = 8
_MAX_K = 50


def _normalize_k(k: Optional[int]) -> int:
    value = _DEFAULT_K if k is None else int(k)
    if value < 1:
        return 1
    if value > _MAX_K:
        return _MAX_K
    return value


def _resolve_published(instance: str):
    """Descriptor for a *published* instance (exact name or slug), else None.
    Mirrors the route: an unknown or draft instance is not retrievable."""
    for cand in (instance, instances.slugify(instance)):
        desc = store.descriptor(cand)
        if desc is not None and desc.status == "published":
            return desc
    return None


def retrieve_chunks(query: str, *, instance: str, k: Optional[int] = None,
                    category: Optional[str] = None) -> dict:
    """Return an ``aiar.retrieve.v1`` result for ``query`` against ``instance``.

    Raises :class:`aiar.contracts.retrieve.EmptyQuery` for a blank query and
    :class:`~aiar.contracts.retrieve.UnknownInstance` for an unknown/unpublished
    instance. Propagates ``store.StoreNotReady`` when the store/embedder cannot
    serve a read. Never invokes a generation model.
    """
    q = (query or "").strip()
    if not q:
        raise EmptyQuery("q is required")
    desc = _resolve_published(instance)
    if desc is None:
        raise UnknownInstance(instance)
    top_k = _normalize_k(k)
    store.ensure_readable()
    category_filter = (category or "").strip() or None
    where = {"category": category_filter} if category_filter else None
    hits = store.query_scored(q, n_results=top_k, where=where, instance=desc.name)
    return serialize_retrieve_result(instance=desc.name, query=q, k=top_k, hits=hits)
