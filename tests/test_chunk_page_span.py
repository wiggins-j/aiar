"""Tests for per-chunk page_span metadata (F013 source-jump prep).

PDF chunks must carry a ``page_span`` (start, end) tuple covering the page
numbers of the paragraphs in that chunk. Non-PDF chunks (markdown, json,
plain text) must keep ``page_span=None``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from aiar.rag.ingest import (
    Chunk,
    _chunk_text,
    ingest_file,
)


def _build_minimal_pdf(pages_text: List[str]) -> bytes:
    """Build a syntactically valid PDF with one text stream per page.

    Uses the built-in Helvetica font (no embedded font data needed) and a
    single BT/ET text-show command per page so pypdf's ``extract_text`` will
    return the supplied string. Intentionally minimal — enough for the
    extractor to round-trip our marker strings, not pretty.
    """
    objects: List[bytes] = []

    def add(obj_bytes: bytes) -> int:
        objects.append(obj_bytes)
        return len(objects)  # 1-indexed object number

    # Reserve object numbers in a stable order: catalog, pages, font,
    # then for each page: page, content. We add them in dependency order.
    # We'll fill in pages-kids and content lengths as we go.

    font_obj_num = None  # filled after we add it
    catalog_obj_num: int
    pages_obj_num: int

    # Build content streams first so we know their lengths.
    content_streams: List[bytes] = []
    for text in pages_text:
        # Escape parentheses and backslashes per PDF string literal rules.
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = (
            b"BT /F1 12 Tf 72 720 Td ("
            + safe.encode("latin-1", errors="replace")
            + b") Tj ET"
        )
        content_streams.append(stream)

    # Order of objects we will emit:
    #   1: Catalog
    #   2: Pages
    #   3: Font
    #   4..: per page (page object, content object) alternating
    # We construct payloads referencing the right object numbers up front.
    catalog_obj_num = 1
    pages_obj_num = 2
    font_obj_num = 3

    page_obj_nums: List[int] = []
    content_obj_nums: List[int] = []
    next_num = 4
    for _ in pages_text:
        page_obj_nums.append(next_num)
        content_obj_nums.append(next_num + 1)
        next_num += 2

    kids_str = " ".join(f"{n} 0 R" for n in page_obj_nums).encode("ascii")

    # Now build the actual object bodies in order.
    add(b"<< /Type /Catalog /Pages " + str(pages_obj_num).encode() + b" 0 R >>")
    add(
        b"<< /Type /Pages /Kids [" + kids_str + b"] /Count "
        + str(len(pages_text)).encode() + b" >>"
    )
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_num, content_num, stream in zip(
        page_obj_nums, content_obj_nums, content_streams
    ):
        page_body = (
            b"<< /Type /Page /Parent " + str(pages_obj_num).encode()
            + b" 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 " + str(font_obj_num).encode()
            + b" 0 R >> >> /Contents " + str(content_num).encode() + b" 0 R >>"
        )
        add(page_body)
        content_body = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"\nendstream"
        )
        add(content_body)

    # Now assemble the PDF bytes with a proper xref table.
    out = bytearray()
    out += b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets: List[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode()
        + b" /Root " + str(catalog_obj_num).encode() + b" 0 R >>\n"
        b"startxref\n" + str(xref_offset).encode() + b"\n%%EOF\n"
    )
    return bytes(out)


@pytest.fixture()
def three_page_pdf(tmp_path: Path) -> Path:
    """A real, parseable 3-page PDF with distinct text on each page.

    Each page contains enough text that, given the chunker's ~1600 char
    target, we still expect multiple chunks across the document — we pad
    each page with repeated paragraphs so the result spans >= 2 chunks.
    """
    para = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "Sphinx of black quartz, judge my vow. "
    ) * 8  # ~ a few hundred chars per page-paragraph
    pages = [
        f"PAGEONE-MARKER {para}",
        f"PAGETWO-MARKER {para}",
        f"PAGETHREE-MARKER {para}",
    ]
    pdf_bytes = _build_minimal_pdf(pages)
    p = tmp_path / "three_pages.pdf"
    p.write_bytes(pdf_bytes)
    return p


def test_pdf_chunks_carry_page_span(three_page_pdf: Path) -> None:
    chunks = ingest_file(three_page_pdf)
    assert chunks, "expected at least one chunk from the 3-page PDF"
    # Every chunk must have a page_span set.
    for c in chunks:
        assert c.page_span is not None, f"chunk {c.chunk_index} missing page_span"
        start, end = c.page_span
        assert isinstance(start, int) and isinstance(end, int)
        assert 1 <= start <= end <= 3, f"page_span {c.page_span} out of range"
    # First chunk should start at page 1; last chunk should reach page 3.
    assert chunks[0].page_span[0] == 1
    assert chunks[-1].page_span[1] == 3
    # Spans should not regress: start of chunk N >= start of chunk N-1.
    starts = [c.page_span[0] for c in chunks]
    assert starts == sorted(starts), f"chunk start pages regressed: {starts}"


def test_pdf_produces_multiple_chunks(three_page_pdf: Path) -> None:
    chunks = ingest_file(three_page_pdf)
    assert len(chunks) >= 2, (
        f"expected >=2 chunks from a padded 3-page PDF, got {len(chunks)}"
    )


def test_markdown_chunks_have_no_page_span(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text(
        "# Heading\n\n"
        "First paragraph of plain markdown content.\n\n"
        "Second paragraph with more text so we still produce a real chunk.\n",
        encoding="utf-8",
    )
    chunks = ingest_file(p)
    assert chunks
    for c in chunks:
        assert c.page_span is None, (
            f"markdown chunk {c.chunk_index} unexpectedly had page_span={c.page_span!r}"
        )


def test_json_chunks_have_no_page_span(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    p.write_text('{"name": "Acme", "window_days": 30}', encoding="utf-8")
    chunks = ingest_file(p)
    assert chunks
    for c in chunks:
        assert c.page_span is None


def test_chunk_text_without_page_nums_yields_none() -> None:
    chunks = _chunk_text(
        source="x", title="x", text="alpha\n\nbeta\n\ngamma", category="general"
    )
    assert chunks
    for c in chunks:
        assert c.page_span is None


def test_chunk_text_with_aligned_page_nums_computes_span() -> None:
    # Three paragraphs on pages 1, 2, 3 — all small enough to land in one chunk,
    # so the chunk's page_span should be (1, 3).
    text = "alpha\n\nbeta\n\ngamma"
    chunks = _chunk_text(
        source="x", title="x", text=text, category="general",
        page_nums=[1, 2, 3],
    )
    assert chunks
    assert chunks[0].page_span == (1, 3)


def test_chunk_dataclass_default_page_span_is_none() -> None:
    # Backward-compat: constructing a Chunk without page_span must work.
    c = Chunk(source="s", title="t", chunk_index=0, text="hi")
    assert c.page_span is None
