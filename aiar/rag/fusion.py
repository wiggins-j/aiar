"""Reciprocal Rank Fusion (RRF) of dense vector + sparse BM25 hits.

RRF is parameter-light and score-scale-agnostic: it fuses by *rank*, not by raw
score, so the incomparable cosine-similarity and BM25 score scales don't need
normalising. ``score(id) = sum over lists of 1 / (k + rank)``, rank 0-based.

Deterministic: ties broken by chunk id so identical inputs give identical order.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

from aiar.rag import store
from aiar.rag.store import RetrievedChunk


def rrf_fuse(
    vector_hits: List[RetrievedChunk],
    bm25_hits: List[Tuple[str, float]],
    *,
    k: int = 60,
    top_n: int,
    instance: "str | None" = None,
) -> List[RetrievedChunk]:
    """Fuse vector + BM25 results by reciprocal rank; return top_n RetrievedChunks.

    The fused RRF value is written into each chunk's ``score`` (higher = better).
    Vector chunks supply text+metadata directly; BM25-only ids are materialised
    via ``store.get_by_ids`` — scoped to ``instance`` (coupling 5b: no
    cross-instance global-collection read).
    """
    rrf: Dict[str, float] = {}
    for rank, hit in enumerate(vector_hits):
        rrf[hit.id] = rrf.get(hit.id, 0.0) + 1.0 / (k + rank)
    for rank, (cid, _score) in enumerate(bm25_hits):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)

    if not rrf:
        return []

    ordered_ids = sorted(rrf, key=lambda cid: (-rrf[cid], cid))[:top_n]

    have: Dict[str, RetrievedChunk] = {h.id: h for h in vector_hits}
    missing = [cid for cid in ordered_ids if cid not in have]
    for chunk in store.get_by_ids(missing, instance=instance):
        have[chunk.id] = chunk

    out: List[RetrievedChunk] = []
    for cid in ordered_ids:
        chunk = have.get(cid)
        if chunk is not None:
            out.append(replace(chunk, score=rrf[cid]))
    return out
