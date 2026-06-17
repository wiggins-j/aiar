"""Unit tests for in-memory document ingest (the remote API server-side path).

Pure-python (chunker only) — no chromadb / embedder needed.
"""
from __future__ import annotations

from aiar.rag import ingest


def test_flat_text_has_no_page_span():
    chunks = ingest.ingest_document(
        source="https://example/doc", title="Doc", text="hello world\n\nsecond para")
    assert chunks
    assert all(c.page_span is None for c in chunks)
    assert all(c.source == "https://example/doc" for c in chunks)
    assert all(c.metadata.get("document_hash") for c in chunks)


def test_pages_round_trip_to_page_span():
    # Big enough paragraphs to force >1 chunk so spans are meaningful.
    pages = [
        {"page": 1, "text": "alpha " * 200},
        {"page": 2, "text": "bravo " * 200},
        {"page": 3, "text": "charlie " * 200},
    ]
    chunks = ingest.ingest_document(source="s", title="t", pages=pages)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.page_span is not None
        lo, hi = c.page_span
        assert 1 <= lo <= hi <= 3


def test_pages_win_over_text_when_both_given():
    chunks = ingest.ingest_document(
        source="s", title="t", text="ignored", pages=[{"page": 5, "text": "real content here"}])
    assert chunks
    assert chunks[0].page_span == (5, 5)


def test_empty_document_returns_no_chunks():
    assert ingest.ingest_document(source="s", title="t", text="   ") == []
    assert ingest.ingest_document(source="s", title="t", pages=[]) == []


def test_malformed_page_numbers_degrade_to_no_span():
    # missing 'page' key must NOT produce page-0 spans — ingest text, span=None
    chunks = ingest.ingest_document(source="s", title="t", pages=[
        {"text": "alpha content"}, {"page": 2, "text": "bravo content"}])
    assert chunks
    assert all(c.page_span is None for c in chunks)
    assert "alpha content" in chunks[0].text


def test_invalid_page_number_below_one_degrades():
    chunks = ingest.ingest_document(source="s", title="t", pages=[
        {"page": 0, "text": "zero page text"}])
    assert chunks
    assert chunks[0].page_span is None


def test_document_hash_stable_and_metadata_carried():
    a = ingest.ingest_document(source="s", title="t", text="same body",
                               metadata={"k": "v"})
    b = ingest.ingest_document(source="s", title="t", text="same body",
                               metadata={"k": "v"})
    assert a[0].metadata["document_hash"] == b[0].metadata["document_hash"]
    assert a[0].metadata["k"] == "v"
    # different text -> different hash
    c = ingest.ingest_document(source="s", title="t", text="other body")
    assert c[0].metadata["document_hash"] != a[0].metadata["document_hash"]


def test_metadata_not_mutated_across_chunks():
    meta = {"shared": 1}
    body = "\n\n".join("para %d %s" % (i, "x " * 200) for i in range(6))
    chunks = ingest.ingest_document(source="s", title="t", text=body, metadata=meta)
    assert len(chunks) >= 2
    # caller's dict is untouched (no document_hash leaked back into it)
    assert meta == {"shared": 1}
