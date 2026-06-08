"""Generic document ingestion: a folder of YOUR documents -> chunks.

This is the model-agnostic, domain-agnostic ingest core. It reads plain
documents from a directory and splits them into overlapping chunks ready for
embedding. Supported file types:

  * ``.txt`` / ``.md`` / ``.markdown`` / ``.rst``  — treated as plain text
  * ``.json``                                       — see :func:`_json_to_text`
  * ``.pdf``                                        — extracted with ``pypdf``

There is NO web-fetching and NO site-specific HTML parsing here — point it at
your own files. (If you want to ingest a website, fetch it to ``.txt``/``.md``
first with a tool of your choice, then ingest the folder.)

Usage:
    python -m aiar.rag.ingest ./examples/docs            # ingest a folder
    python -m aiar.rag.ingest ./examples/docs --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ~400 tokens at ~4 chars/token, with a small overlap so a fact split across a
# chunk boundary still survives in at least one chunk.
_CHUNK_TARGET_CHARS = 1600
_CHUNK_OVERLAP_CHARS = 200

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".text"}
_JSON_SUFFIXES = {".json"}
_PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = _TEXT_SUFFIXES | _JSON_SUFFIXES | _PDF_SUFFIXES


@dataclass
class Chunk:
    """One ingestable chunk plus retrieval metadata.

    ``source`` is the document path (or other provenance id), ``title`` is a
    human label (defaults to the filename), ``category`` is a free-form tag you
    may use for ``where=`` metadata filtering at query time.
    """

    source: str
    title: str
    chunk_index: int
    text: str
    category: str = "general"
    metadata: Dict[str, object] = field(default_factory=dict)
    page_span: Optional[Tuple[int, int]] = None


def _json_to_text(raw: str) -> str:
    """Flatten a JSON document into readable text.

    A list of ``{...}`` records becomes one block per record; a dict becomes
    ``key: value`` lines; anything else is stringified. Deliberately simple and
    generic — bring your own pre-processing for richer shapes.
    """
    try:
        data = json.loads(raw)
    except ValueError:
        return raw

    def render(obj, indent: int = 0) -> str:
        pad = "  " * indent
        if isinstance(obj, dict):
            return "\n".join(f"{pad}{k}: {render(v, indent + 1).lstrip()}" for k, v in obj.items())
        if isinstance(obj, list):
            return "\n\n".join(render(item, indent) for item in obj)
        return f"{pad}{obj}"

    return render(data)


def _pdf_to_text(path: Path) -> List[Tuple[str, int]]:
    """Extract text from a PDF with pypdf.

    Returns a list of ``(page_text, page_num)`` tuples (1-indexed page numbers),
    skipping empty pages. Returns ``[]`` on extraction errors. Page-number
    tracking is structured (no ``[Page N]`` marker strings prepended) so
    downstream chunking can build ``page_span`` metadata cleanly.
    """
    try:
        from pypdf import PdfReader
    except Exception as exc:
        logger.warning("ingest: could not read PDF %s (missing pypdf): %s", path, exc)
        return []
    try:
        reader = PdfReader(str(path))
        pages: List[Tuple[str, int]] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((text, idx))
        return pages
    except Exception as exc:
        logger.warning("ingest: could not extract PDF %s: %s", path, exc)
        return []


def _chunk_text(source: str, title: str, text: str, category: str,
                metadata: "Dict[str, object] | None" = None,
                page_nums: Optional[List[int]] = None) -> List[Chunk]:
    """Split text into overlapping chunks on paragraph boundaries.

    If ``page_nums`` is provided, it must be a list aligned 1:1 with the
    paragraphs derived from ``text`` (split on blank lines). Each emitted
    chunk's ``page_span`` is then ``(min, max)`` of the page numbers of the
    paragraphs included in that chunk. If ``page_nums`` is ``None``, chunks
    have ``page_span=None`` (non-PDF callers).
    """
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if not paragraphs:
        flat = text.strip()
        paragraphs = [flat] if flat else []
        # If we collapsed to a single paragraph but page_nums was per-paragraph
        # of the original split, the alignment is lost — drop it defensively.
        if page_nums is not None and len(page_nums) != len(paragraphs):
            page_nums = None
    # Defensive: if caller supplied a mismatched length, ignore page tracking
    # rather than producing wrong spans.
    if page_nums is not None and len(page_nums) != len(paragraphs):
        page_nums = None
    chunks: List[Chunk] = []
    current = ""
    prev_tail = ""
    current_pages: List[int] = []
    chunk_meta = dict(metadata or {})

    def _span(pages: List[int]) -> Optional[Tuple[int, int]]:
        if not pages:
            return None
        return (min(pages), max(pages))

    for i, para in enumerate(paragraphs):
        para_page = page_nums[i] if page_nums is not None else None
        if len(current) + len(para) > _CHUNK_TARGET_CHARS and current:
            chunks.append(Chunk(
                source=source, title=title, chunk_index=len(chunks),
                text=(prev_tail + current).strip(), category=category,
                metadata=dict(chunk_meta),
                page_span=_span(current_pages) if page_nums is not None else None,
            ))
            prev_tail = current[-_CHUNK_OVERLAP_CHARS:]
            current = ""
            current_pages = []
        current += ("\n\n" if current else "") + para
        if para_page is not None:
            current_pages.append(para_page)
    if current.strip():
        chunks.append(Chunk(
            source=source, title=title, chunk_index=len(chunks),
            text=(prev_tail + current).strip(), category=category,
            metadata=dict(chunk_meta),
            page_span=_span(current_pages) if page_nums is not None else None,
        ))
    return chunks


def ingest_file(path: Path, *, category: str = "general") -> List[Chunk]:
    """Read one document and return its chunks. Returns [] on any error."""
    front_matter = {}
    body = ""
    raw = ""
    if path.suffix.lower() not in _PDF_SUFFIXES:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("ingest: could not read %s: %s", path, exc)
            return []
        body = raw
    if path.suffix.lower() in _TEXT_SUFFIXES:
        try:
            from aiar.rag import metadata as rag_metadata
            front_matter, body = rag_metadata.parse_front_matter(raw)
        except Exception:
            front_matter, body = {}, raw
    page_nums: Optional[List[int]] = None
    if path.suffix.lower() in _JSON_SUFFIXES:
        text = _json_to_text(raw)
    elif path.suffix.lower() in _PDF_SUFFIXES:
        pages = _pdf_to_text(path)
        # Build a text body whose paragraph splits align with page_nums.
        # Each page's text becomes one or more paragraphs (split on blank
        # lines); we record the page number for each paragraph so chunking
        # can later compute (start, end) page_span.
        para_texts: List[str] = []
        para_pages: List[int] = []
        for page_text, page_num in pages:
            page_paragraphs = [
                p.strip()
                for p in page_text.replace("\r\n", "\n").split("\n\n")
                if p.strip()
            ]
            if not page_paragraphs:
                stripped = page_text.strip()
                if stripped:
                    page_paragraphs = [stripped]
            for p in page_paragraphs:
                para_texts.append(p)
                para_pages.append(page_num)
        text = "\n\n".join(para_texts)
        if para_pages:
            page_nums = para_pages
    else:
        text = body
    if not text.strip():
        return []
    doc_meta = dict(front_matter)
    doc_meta["document_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return _chunk_text(str(path), path.stem, text, category,
                       metadata=doc_meta, page_nums=page_nums)


def iter_documents(folder: Path) -> Iterable[Path]:
    """Yield every supported document under ``folder`` (recursive)."""
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p


def ingest_folder(folder: Path, *, category: str = "general") -> List[Chunk]:
    """Read every supported document under ``folder`` and return all chunks."""
    chunks: List[Chunk] = []
    for doc in iter_documents(folder):
        chunks.extend(ingest_file(doc, category=category))
    return chunks


# ------------------------------------------------------------------------ CLI


def _print_dry_run(chunks: List[Chunk]) -> None:
    print(f"DRY RUN — {len(chunks)} chunk(s) would be ingested.\n")
    by_source: "dict[str, List[Chunk]]" = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c)
    for source, items in by_source.items():
        print(f"## {source}  ({len(items)} chunks)")
        for c in items[:3]:
            preview = c.text.replace("\n", " ")[:160]
            print(f"   - [{c.chunk_index}] len={len(c.text)}: {preview}"
                  f"{'...' if len(c.text) > 160 else ''}")
        if len(items) > 3:
            print(f"   ... and {len(items) - 3} more chunks")
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aiar.rag.ingest",
        description="Ingest a folder of documents into the AIAR RAG store.",
    )
    parser.add_argument("folder", help="Path to a folder of .txt/.md/.json/.pdf documents")
    parser.add_argument("--category", default="general",
                        help="Free-form metadata tag for the chunks (default: general)")
    parser.add_argument("--instance", default=None,
                        help="RAG instance to ingest into (default: the active "
                             "instance, i.e. RAG_INSTANCE or 'default'). A new "
                             "named instance is created on first ingest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be ingested without writing to the store")
    parser.add_argument("--validate", default=None, metavar="SCHEMA",
                        help="Validate each document's front-matter against a "
                             "metadata schema JSON before ingesting "
                             "(warnings only; never blocks)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"error: {folder} is not a directory", file=sys.stderr)
        return 2
    if args.validate:
        from aiar.rag import metadata
        try:
            schema = metadata.load_schema(args.validate)
        except (OSError, ValueError) as exc:
            print(f"error: could not load schema {args.validate}: {exc}", file=sys.stderr)
            return 2
        results = metadata.validate_folder(folder, schema)
        bad = {p: iss for p, iss in results.items() if iss}
        if bad:
            print(f"metadata: {len(bad)}/{len(results)} file(s) have schema issues "
                  f"(warning only — ingest continues):", file=sys.stderr)
            for p, iss in bad.items():
                print(f"  {p}: {'; '.join(iss)}", file=sys.stderr)
        else:
            print(f"metadata: all {len(results)} file(s) valid against "
                  f"{schema.get('name', '?')!r}.")
    chunks = ingest_folder(folder, category=args.category)
    if not chunks:
        print("No chunks produced (no supported documents found).", file=sys.stderr)
        return 1
    if args.dry_run:
        _print_dry_run(chunks)
        return 0
    from aiar.rag import store
    store.init()
    if args.instance:
        instance = store.create_instance(args.instance, display_name=args.instance)
    else:
        instance = store.active_instance()
    if instance == store.NO_RAG:
        print(f"error: cannot ingest into the No-RAG instance ({store.NO_RAG!r}); "
              f"pick a real instance with --instance or RAG_INSTANCE",
              file=sys.stderr)
        return 2
    written = store.add(chunks, instance=instance)
    # A corpus with content is no longer a draft — publish it so the watcher GUI
    # shows it as ready instead of perpetually "[draft]".
    if (store.chunk_count(instance=instance) or 0) > 0:
        try:
            store.publish_instance(instance)
        except Exception as exc:
            logger.warning("ingest: could not publish instance %r: %s", instance, exc)
    print(f"Wrote {written} new chunks ({len(chunks)} candidates) "
          f"to instance {instance!r}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
