"""Cross-encoder reranking of a wide first-pass candidate set.

Bi-encoder cosine (``all-MiniLM-L6-v2``) is cheap but coarse: the chunk that
actually answers a question can rank below near-duplicates. A cross-encoder
scores the query and each candidate *jointly*, so it reorders far more
accurately. We pull a wide first pass (``RAG_FETCH_K``) and keep the best
``top_k``. Ranking-only — no re-ingest, no corpus change.

The model is loaded once (lazy singleton) and held CPU-resident. Any failure
degrades to the input order so retrieval never breaks on the reranker.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import replace
from typing import List

from aiar.rag.store import RetrievedChunk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None
_lock = threading.Lock()


def _get_model():
    """Lazily load + cache the CrossEncoder singleton (thread-safe)."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import CrossEncoder

            name = os.environ.get("RAG_RERANK_MODEL", _DEFAULT_MODEL)
            logger.info("reranker: loading cross-encoder %s", name)
            _model = CrossEncoder(name)
    return _model


def rerank(query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
    """Rescore candidates with the cross-encoder; return the best top_k.

    The cross-encoder score replaces each chunk's ``score`` (higher = better).
    On any failure, falls back to the input order (``candidates[:top_k]``).
    """
    if not candidates or top_k <= 0:
        return []
    try:
        model = _get_model()
        start = time.monotonic()
        scores = model.predict([(query, c.text) for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        out = [replace(candidates[i], score=float(scores[i])) for i in order[:top_k]]
        logger.debug("reranker: %d->%d candidates in %dms",
                     len(candidates), len(out), int((time.monotonic() - start) * 1000))
        return out
    except Exception as exc:
        logger.error("reranker: failed (%s) — falling back to input order", exc)
        return candidates[:top_k]
