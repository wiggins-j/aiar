"""Synchronous document ingest — the in-process twin of
``POST /instances/{instance}/documents``.

Shares the per-document loop (``apply_documents``) and the ``aiar.ingest.v1``
result serializer with the HTTP route, so a synchronous Python caller and a
remote caller that polls the job to completion get identical shapes. ``publish``
defaults to ``False`` (fail-closed): publishing is an explicit step, never a side
effect of ingest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from aiar.contracts.ingest import (
    UnknownInstance,
    serialize_ingest_result,
    status_for,
)
from aiar.rag import ingest as _ingest
from aiar.rag import instances
from aiar.rag import store as _store


@dataclass
class _Fields:
    doc_id: str
    source: str
    title: str
    category: str
    text: Optional[str]
    pages: Optional[list]
    metadata: Optional[dict]


def _fields(doc: Any) -> _Fields:
    """Read document fields from either a dict or an attribute-bearing object
    (e.g. the route's pydantic ``DocumentIn``)."""
    if isinstance(doc, dict):
        get = lambda k, d=None: doc.get(k, d)  # noqa: E731
    else:
        get = lambda k, d=None: getattr(doc, k, d)  # noqa: E731
    return _Fields(
        doc_id=str(get("doc_id", "") or ""),
        source=str(get("source", "") or ""),
        title=get("title", "") or "",
        category=get("category", "general") or "general",
        text=get("text", None),
        pages=get("pages", None),
        metadata=get("metadata", None),
    )


def apply_documents(name: str, documents: List[Any], *, store, ingest
                    ) -> Tuple[int, int, List[dict]]:
    """Embed + store each document into ``name``. Returns
    ``(chunks_added, duplicates, errors)``. One bad document never aborts the
    batch — it lands in ``errors`` and the loop continues.

    ``store`` / ``ingest`` are injected so the HTTP route can pass the references
    its tests monkeypatch; the twin passes the real modules.
    """
    chunks_added = 0
    duplicates = 0
    errors: List[dict] = []
    for doc in documents:
        f = _fields(doc)
        try:
            chunks = ingest.ingest_document(
                source=f.source, title=f.title or f.source,
                text=f.text, pages=f.pages, category=f.category,
                metadata={**(f.metadata or {}), "doc_id": f.doc_id})
            if not chunks:
                errors.append({"doc_id": f.doc_id, "error": "no usable text"})
                continue
            # store.add dedups by document_hash: a re-posted document adds 0.
            added = store.add(chunks, instance=name)
            chunks_added += added
            duplicates += len(chunks) - added
        except Exception as exc:  # one bad doc never aborts the batch
            errors.append({"doc_id": f.doc_id, "error": str(exc)})
    return chunks_added, duplicates, errors


def last_error(errors: List[dict]) -> Optional[str]:
    """The last non-empty error message in a batch (for last_ingest_error)."""
    msg = None
    for e in errors:
        if e.get("error"):
            msg = e["error"]
    return msg


def record_ingest_safe(store, name: str, errors: List[dict]) -> None:
    """Best-effort write of last-ingest readiness state. Must never break an
    ingest — the store double in tests may not implement it."""
    fn = getattr(store, "record_ingest", None)
    if fn is None:
        return
    try:
        fn(name, error=last_error(errors))
    except Exception:  # pragma: no cover - defensive
        pass


def _resolve_desc(instance: str):
    for cand in (instance, instances.slugify(instance)):
        desc = _store.descriptor(cand)
        if desc is not None:
            return desc
    return None


def ingest_documents(documents, *, instance: str, publish: bool = False) -> dict:
    """Ingest ``documents`` into ``instance`` synchronously; return an
    ``aiar.ingest.v1`` result.

    Raises :class:`aiar.contracts.ingest.UnknownInstance` for an unknown instance
    and propagates ``store.StoreNotReady`` when the store/embedder can't write.
    Ingest may target a draft instance (you ingest before publishing).
    """
    documents = list(documents)
    _store.ensure_writable()
    desc = _resolve_desc(instance)
    if desc is None:
        raise UnknownInstance(instance)
    name = desc.name

    chunks_added, duplicates, errors = apply_documents(
        name, documents, store=_store, ingest=_ingest)

    published = getattr(desc, "status", None) == "published"
    publish_failed = False
    if publish and not (chunks_added == 0 and errors):
        try:
            _store.publish_instance(name)
            published = True
        except Exception as exc:
            publish_failed = True
            errors.append({"doc_id": None, "error": f"publish failed: {exc}"})

    status = status_for(chunks_added=chunks_added, errors=errors,
                        publish_failed=publish_failed)
    record_ingest_safe(_store, name, errors)
    return serialize_ingest_result(
        instance=name, status=status, accepted=len(documents),
        chunks_added=chunks_added, duplicates=duplicates,
        errors=errors, published=published)
