"""AIAR RAG pipeline: ingest -> store -> retrieve (hybrid + rerank + rewrite).

Public surface:
    - ``ingest``      : turn a folder of .txt/.md/.json documents into chunks
    - ``store``       : ChromaDB vector store (add / query_scored / get_by_ids)
    - ``retriever``   : ``get_context(query)`` — the single retrieval entrypoint
    - ``lexical``     : BM25 sparse index over the corpus
    - ``fusion``      : reciprocal-rank fusion of vector + BM25
    - ``reranker``    : cross-encoder reranking of a wide candidate set
    - ``query_rewrite``: optional pre-retrieval rewrite / HyDE
"""

from . import store, retriever, ingest, lexical, fusion, reranker, query_rewrite
from .store import RetrievedChunk
from .ingest import Chunk

__all__ = [
    "store",
    "retriever",
    "ingest",
    "lexical",
    "fusion",
    "reranker",
    "query_rewrite",
    "RetrievedChunk",
    "Chunk",
]
