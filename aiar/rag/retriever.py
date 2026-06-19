"""Retrieve relevant context for a free-form query.

This is the single retrieval entrypoint. It composes the full pipeline:

    query
      -> optional pre-retrieval rewrite / HyDE   (RAG_QUERY_REWRITE_MODE)
      -> candidate retrieval:
           hybrid  (vector + BM25, RRF-fused)    (RAG_HYBRID_ENABLED)
           or plain vector
      -> optional cross-encoder rerank           (RAG_RERANK_ENABLED)
      -> top-k chunks -> labelled context block

Every advanced stage is behind an env flag and defaults OFF, so the bare path
is identical to plain vector retrieval. Turn flags on to trade a little latency
for relevance. All failures degrade gracefully to an empty string — the caller
(the harness) treats no-context as "answer from the bare model".
"""
from __future__ import annotations

import logging
import os
from typing import List

from aiar.rag import store
from aiar.rag import settings as rag_settings

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _flag(name: str) -> bool:
    raw = os.environ.get(name)
    return raw is not None and raw.strip().lower() in ("1", "true", "yes", "on")


def hybrid_enabled() -> bool:
    """Effective hybrid flag (live override -> ``RAG_HYBRID_ENABLED`` -> False)."""
    return bool(rag_settings.get("hybrid"))


def rerank_enabled() -> bool:
    """Effective rerank flag (live override -> ``RAG_RERANK_ENABLED`` -> False)."""
    return bool(rag_settings.get("rerank"))


# "No RAG" sentinel: selecting this instance skips retrieval entirely (mirrors
# rag=false answerer-blinding). The store never gets a ``none`` handle.
NO_RAG = "none"


def _retrieve_candidates(query: str, n: int, *, where: "dict | None" = None,
                         instance: "str | None" = None
                         ) -> List[store.RetrievedChunk]:
    """Top-n scored candidates: hybrid (BM25+vector RRF) when enabled, else vector."""
    if where is not None:
        # The lexical/BM25 index is text-only and cannot enforce Chroma metadata
        # filters. For filtered retrieval, stay on the vector path so a
        # ``where=`` constraint cannot be diluted by unfiltered lexical hits.
        return store.query_scored(query, n, where=where, instance=instance)
    if hybrid_enabled():
        from aiar.rag import lexical, fusion
        vector = store.query_scored(query, _env_int("RAG_VECTOR_K", 20),
                                    where=where, instance=instance)
        idx = lexical.index(instance=instance)
        bm25 = idx.search(query, _env_int("RAG_BM25_K", 20)) if idx is not None else []
        return fusion.rrf_fuse(vector, bm25, k=_env_int("RAG_RRF_K", 60),
                               top_n=n, instance=instance)
    return store.query_scored(query, n, where=where, instance=instance)


def _retrieve_texts(query: str, k: int, *, rerank: bool, where: "dict | None" = None,
                    instance: "str | None" = None
                    ) -> List[str]:
    """Top-k chunk texts. retrieve candidates (hybrid | vector) -> optional
    cross-encoder rerank -> top-k. With rerank, hybrid AND filter all off this
    is byte-identical to the plain vector path."""
    if not rerank and not hybrid_enabled() and where is None:
        return store.query(query, n_results=k, instance=instance)
    n = int(rag_settings.get("fetch_k")) if rerank else k
    cands = _retrieve_candidates(query, max(n, k), where=where, instance=instance)
    if rerank:
        from aiar.rag import reranker
        cands = reranker.rerank(query, cands, k)
    else:
        cands = cands[:k]
    return [c.text for c in cands]


def _retrieve_chunk_objs(query: str, k: int, *, rerank: bool,
                         where: "dict | None" = None,
                         instance: "str | None" = None
                         ) -> List[store.RetrievedChunk]:
    """Top-k chunk OBJECTS actually used (post hybrid/rerank/truncate). Same
    selection as ``_retrieve_texts`` but returns the scored chunks so a caller
    can report the answerer's real source set (``answer_prompt`` sources)."""
    n = int(rag_settings.get("fetch_k")) if rerank else k
    cands = _retrieve_candidates(query, max(n, k), where=where, instance=instance)
    if rerank:
        from aiar.rag import reranker
        return reranker.rerank(query, cands, k)
    return cands[:k]


def get_context(query: str, *, instance: "str | None" = None,
                top_k: "int | None" = None,
                rerank: "bool | None" = None,
                where: "dict | None" = None,
                rewrite: bool = True,
                capture: "dict | None" = None) -> str:
    """Return a labelled context block for a free-form ``query``.

    Returns an empty string if RAG is unavailable, the store is empty, the
    query is blank, ``instance == "none"`` (No RAG selected), or retrieval
    raises (graceful degradation — the harness relies on this).

    ``instance``: which named RAG instance to retrieve from (None -> resolves to
                  the process-active instance -> default). ``"none"`` skips
                  retrieval entirely (mirrors rag=false).
    ``top_k``  : number of chunks (default ``RAG_TOP_K`` env, else 3).
    ``rerank`` : override the global ``RAG_RERANK_ENABLED`` flag. None -> flag.
    ``where``  : optional ChromaDB metadata filter (e.g. {"category": "faq"}).
    ``rewrite``: when True, honour ``RAG_QUERY_REWRITE_MODE`` (off -> no-op);
                 pass False to skip the extra rewrite LLM call entirely.
    ``capture``: optional dict; when given, ``capture["chunks"]`` is set to the
                 list of ``RetrievedChunk`` actually used (for answer sources).
    """
    if instance == NO_RAG:
        return ""
    # Resolve the active instance: if "No RAG" is the active selection, an
    # instance-less call must also skip retrieval.
    if instance is None:
        try:
            if store.active_instance() == NO_RAG:
                return ""
        except Exception:
            pass
    if not query or not query.strip():
        return ""
    k = top_k if top_k is not None else int(rag_settings.get("top_k"))
    do_rerank = rerank_enabled() if rerank is None else rerank

    if rewrite:
        from aiar.rag import query_rewrite
        retrieval_query = query_rewrite.transform(query, instance=instance)
    else:
        retrieval_query = query

    try:
        if capture is not None:
            objs = _retrieve_chunk_objs(retrieval_query, k, rerank=do_rerank,
                                        where=where, instance=instance)
            capture["chunks"] = objs
            chunks = [c.text for c in objs]
        else:
            chunks = _retrieve_texts(retrieval_query, k, rerank=do_rerank,
                                     where=where, instance=instance)
    except Exception as exc:
        logger.debug("get_context: query failed: %s", exc)
        if capture is not None:
            capture.setdefault("chunks", [])
        return ""

    if not chunks:
        return ""
    body = "\n\n".join(f"[chunk {i + 1}]\n{c}" for i, c in enumerate(chunks))
    return f"--- Knowledge (top-{len(chunks)}) ---\n{body}\n--- End Knowledge ---"
