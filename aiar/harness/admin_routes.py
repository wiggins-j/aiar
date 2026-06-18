"""Authenticated remote ingest, retrieve + instance-management routes for AIAR.

Mounted on the harness ``app`` (see ``service.py``). Every route requires the
bearer token (``require_token`` at router level) and the mutating ones gate on
``store.ensure_writable()`` so a remote client can never get a false "success"
when nothing was embedded. All embedding is server-side MiniLM — no route
accepts client vectors.

This module is intentionally thin: it wraps existing ``aiar.rag.store`` /
``aiar.rag.ingest`` functions in HTTP. No new retrieval logic.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from aiar.harness import ingest_jobs
from aiar.harness.auth import require_token
from aiar.rag import ingest, instances, store

# Request caps: clients should page above these.
_MAX_DOCS = 200
_MAX_BYTES = 8 * 1024 * 1024  # 8 MB of document text per request
_RETRIEVE_DEFAULT_K = 8
_RETRIEVE_MAX_K = 50

router = APIRouter(dependencies=[Depends(require_token)])


# --- models -----------------------------------------------------------------

class CreateInstance(BaseModel):
    name: str = Field(min_length=1)
    display_name: Optional[str] = None
    query_rewrite: Optional[Dict[str, str]] = None
    rerank_model: Optional[str] = None


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1)              # idempotency key (sha256 etc.)
    source: str = Field(min_length=1)              # provenance -> Chunk.source
    title: str = ""
    category: str = "general"
    text: Optional[str] = None                     # flat body (page_span=None)
    pages: Optional[List[Dict[str, Any]]] = None   # [{page, text}] -> page_span
    metadata: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    documents: List[DocumentIn]
    publish: bool = False


class RetrieveRequest(BaseModel):
    q: str
    k: Optional[int] = None
    category: Optional[str] = None


# --- helpers ----------------------------------------------------------------

def _resolve_desc(instance: str):
    """Descriptor for an already-existing instance (matched by exact name or
    slug), or None. Reuses ``store.descriptor`` so resolution matches the rest of
    AIAR and avoids ``list_instances``' per-instance ``chunk_count`` cost. We
    never auto-create on these routes: an unknown name is a 404,
    not a stray corpus."""
    for cand in (instance, instances.slugify(instance)):
        desc = store.descriptor(cand)
        if desc is not None:
            return desc
    return None


def _require_writable() -> None:
    try:
        store.ensure_writable()
    except store.StoreNotReady as exc:
        raise HTTPException(status_code=503,
                            detail={"code": exc.code, "error": str(exc)})


def _require_readable() -> None:
    try:
        store.ensure_readable()
    except store.StoreNotReady as exc:
        raise HTTPException(status_code=503,
                            detail={"code": exc.code, "error": str(exc)})


def _doc_bytes(doc: DocumentIn) -> int:
    n = len((doc.text or "").encode("utf-8"))
    for p in (doc.pages or []):
        n += len(str(p.get("text") or "").encode("utf-8"))
    return n


def _normalize_retrieve_k(k: Optional[int]) -> int:
    value = _RETRIEVE_DEFAULT_K if k is None else int(k)
    if value < 1:
        return 1
    if value > _RETRIEVE_MAX_K:
        return _RETRIEVE_MAX_K
    return value


def _parse_page_span(value: Any) -> Optional[List[int]]:
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


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _retrieve_hit_to_dict(hit) -> dict:
    meta = dict(hit.metadata or {})
    return {
        "chunk_id": hit.id,
        "source": str(meta.get("source") or ""),
        "title": str(meta.get("title") or ""),
        "text": hit.text,
        "score": hit.score,
        "chunk_index": _optional_int(meta.get("index")),
        "category": str(meta.get("category") or "general"),
        "page_span": _parse_page_span(meta.get("page_span")),
        "metadata": meta,
    }


def _retrieve_response(instance: str, q: str, k: Optional[int],
                       category: Optional[str]) -> dict:
    query = (q or "").strip()
    if not query:
        raise HTTPException(400, {"code": "empty_query", "error": "q is required"})
    desc = _resolve_desc(instance)
    if desc is None or desc.status != "published":
        raise HTTPException(404, {"code": "unknown_instance", "error": instance})
    top_k = _normalize_retrieve_k(k)
    _require_readable()
    category_filter = (category or "").strip() or None
    where = {"category": category_filter} if category_filter else None
    hits = store.query_scored(query, n_results=top_k, where=where, instance=desc.name)
    out = [_retrieve_hit_to_dict(h) for h in hits]
    return {
        "instance": desc.name,
        "query": query,
        "k": top_k,
        "score_kind": "cosine_similarity",
        "score_order": "desc",
        "hits": out,
        "count": len(out),
    }


# --- instance management ----------------------------------------------------

@router.post("/instances")
def create_instance(req: CreateInstance) -> dict:
    existed = store.descriptor(instances.slugify(req.name)) is not None
    slug = store.create_instance(
        req.name, display_name=req.display_name,
        query_rewrite=req.query_rewrite, rerank_model=req.rerank_model)
    desc = store.descriptor(slug)
    status = desc.status if desc is not None else "draft"
    return {"instance": slug, "status": status, "created": not existed}


@router.get("/instances")
def list_instances() -> dict:
    out = store.list_instances()
    for d in out:
        d["published"] = d.get("status") == "published"
    return {"instances": out}


@router.get("/instances/{instance}/health")
def instance_health(instance: str) -> dict:
    desc = _resolve_desc(instance)
    if desc is None:
        raise HTTPException(404, {"code": "unknown_instance", "error": instance})
    h = store.health(instance=desc.name)
    h["published"] = desc.status == "published"
    return h


@router.get("/instances/{instance}/retrieve")
def retrieve_instance(instance: str, q: str = Query(...),
                      k: Optional[int] = Query(default=None),
                      category: Optional[str] = Query(default=None)) -> dict:
    return _retrieve_response(instance, q, k, category)


@router.post("/instances/{instance}/retrieve")
def retrieve_instance_body(instance: str, req: RetrieveRequest) -> dict:
    return _retrieve_response(instance, req.q, req.k, req.category)


@router.post("/instances/{instance}/publish")
def publish_instance(instance: str) -> dict:
    desc = _resolve_desc(instance)
    if desc is None:
        raise HTTPException(404, {"code": "unknown_instance", "error": instance})
    store.publish_instance(desc.name)
    return {"instance": desc.name, "published": True}


@router.delete("/instances/{instance}")
def delete_instance(instance: str) -> dict:
    desc = _resolve_desc(instance)
    if desc is None:
        raise HTTPException(404, {"code": "unknown_instance", "error": instance})
    try:
        result = store.delete_instance(desc.name)
    except ValueError as exc:  # default / reserved / none
        raise HTTPException(400, {"code": "protected", "error": str(exc)})
    return {"deleted": True, "active": result.get("active")}


# --- document ingest --------------------------------------------------------

@router.post("/instances/{instance}/documents", status_code=202)
def ingest_documents(instance: str, req: IngestRequest) -> dict:
    _require_writable()
    desc = _resolve_desc(instance)
    if desc is None:
        raise HTTPException(404, {"code": "unknown_instance", "error": instance})
    name = desc.name
    if len(req.documents) > _MAX_DOCS:
        raise HTTPException(413, {"code": "too_many_documents",
                                  "error": f"max {_MAX_DOCS} documents per request"})
    if sum(_doc_bytes(d) for d in req.documents) > _MAX_BYTES:
        raise HTTPException(413, {"code": "payload_too_large",
                                  "error": f"max {_MAX_BYTES} bytes of text per request"})

    job = ingest_jobs.new_job(name, documents_total=len(req.documents))
    publish_failed = False
    for doc in req.documents:
        try:
            chunks = ingest.ingest_document(
                source=doc.source, title=doc.title or doc.source,
                text=doc.text, pages=doc.pages, category=doc.category,
                metadata={**(doc.metadata or {}), "doc_id": doc.doc_id})
            if not chunks:
                job.errors.append({"doc_id": doc.doc_id, "error": "no usable text"})
                continue
            # Idempotency lives in store.add: it skips re-embedding chunks whose
            # document_hash + count are unchanged, so a re-posted document adds 0.
            added = store.add(chunks, instance=name)
            job.chunks_added += added
            job.duplicates += len(chunks) - added
        except Exception as exc:  # one bad doc never aborts the batch
            job.errors.append({"doc_id": doc.doc_id, "error": str(exc)})

    # Publish only if requested AND the batch wasn't a total failure — don't flip
    # a brand-new draft to published when nothing landed and docs errored. A
    # re-ingest of all-duplicates (0 added, no errors) is a legitimate publish.
    if req.publish and not (job.chunks_added == 0 and job.errors):
        try:
            store.publish_instance(name)
        except Exception as exc:
            publish_failed = True
            job.errors.append({"doc_id": None, "error": f"publish failed: {exc}"})

    # "failed" if the requested publish failed, or nothing was stored despite
    # errors (never report success when no chunks landed — AC#4). All-duplicates
    # with no errors is a legitimate idempotent no-op, so that stays "done".
    failed = publish_failed or (job.chunks_added == 0 and bool(job.errors))
    ingest_jobs.finish(job, "failed" if failed else "done")
    return {"job_id": job.job_id, "accepted": len(req.documents), "instance": name}


@router.get("/instances/{instance}/ingest-jobs/{job_id}")
def ingest_job_status(instance: str, job_id: str) -> dict:
    desc = _resolve_desc(instance)
    name = desc.name if desc is not None else instance
    job = ingest_jobs.get(name, job_id)
    if job is None:
        raise HTTPException(404, {"code": "unknown_job", "error": job_id})
    return job.to_dict()
