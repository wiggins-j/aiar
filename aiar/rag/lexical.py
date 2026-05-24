"""Sparse BM25 lexical index over the document corpus.

Dense vector retrieval (``store.query_scored``) is strong on paraphrase but
weak on exact tokens — proper nouns, identifiers, numbers, thresholds. BM25
rewards exact-term overlap, so fusing the two (hybrid, see ``rag.fusion``)
recovers literal-term recall.

The index is built once from ``store.all_documents()`` (the single source of
truth — no second copy of the corpus on disk) and held as a module singleton.
A corpus re-ingest auto-invalidates the index via the chunk-count cache key;
``invalidate()`` forces it.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import List, Optional, Tuple

from aiar.rag import store

logger = logging.getLogger(__name__)

# Per-instance cache: instance name -> (BM25Index, chunk_count). Keying on the
# instance (and that instance's chunk_count) keeps each corpus's index isolated —
# a re-ingest of one instance never rebuilds another's.
_indexes: "dict[str, tuple[BM25Index, int]]" = {}
_lock = threading.Lock()

# Lower-case, keep digits, split on non-alphanumeric. Keeping digits matters:
# numbers often carry the answer and must survive tokenisation as their own
# tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """rank_bm25.BM25Okapi over the corpus, aligned to chunk ids by position."""

    def __init__(self, ids: List[str], docs: List[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.ids = ids
        self._tokenized = [tokenize(d) for d in docs]
        self._bm25 = BM25Okapi(self._tokenized)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Return up to k (chunk_id, bm25_score), highest score first."""
        if not self.ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.ids[i], float(scores[i])) for i in order[:k]]


def index(*, instance: "Optional[str]" = None) -> "Optional[BM25Index]":
    """Lazily build and return the BM25 index for ``instance``, or None if that
    instance's corpus is empty.

    Thread-safe (double-checked lock). The cache is keyed on
    ``(instance, store.chunk_count(instance))`` so a re-ingest of one instance
    auto-rebuilds only that instance's index.
    """
    name = store.active_instance() if instance is None else instance
    count = store.chunk_count(instance=name)
    if count is None or count == 0:
        return None
    cached = _indexes.get(name)
    if cached is not None and cached[1] == count:
        return cached[0]
    with _lock:
        cached = _indexes.get(name)
        if cached is not None and cached[1] == count:
            return cached[0]
        ids, docs = store.all_documents(instance=name)
        if not ids:
            return None
        start = time.monotonic()
        try:
            built = BM25Index(ids, docs)
        except Exception as exc:
            logger.error("lexical: BM25 index build failed: %s", exc)
            return None
        _indexes[name] = (built, count)
        logger.info("lexical: BM25 index built (instance=%s, %d chunks, %dms)",
                    name, len(built), int((time.monotonic() - start) * 1000))
        return built


def invalidate(*, instance: "Optional[str]" = None) -> None:
    """Drop the cached index so the next index() rebuilds from the store. With
    no ``instance`` argument, drops every instance's cached index."""
    with _lock:
        if instance is None:
            _indexes.clear()
        else:
            _indexes.pop(instance, None)
