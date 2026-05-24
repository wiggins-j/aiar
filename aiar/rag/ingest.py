"""Generic document ingestion: a folder of YOUR documents -> chunks.

This is the model-agnostic, domain-agnostic ingest core. It reads plain
documents from a directory and splits them into overlapping chunks ready for
embedding. Supported file types:

  * ``.txt`` / ``.md`` / ``.markdown`` / ``.rst``  — treated as plain text
  * ``.json``                                       — see :func:`_json_to_text`

There is NO web-fetching and NO site-specific HTML parsing here — point it at
your own files. (If you want to ingest a website, fetch it to ``.txt``/``.md``
first with a tool of your choice, then ingest the folder.)

Usage:
    python -m aiar.rag.ingest ./examples/docs            # ingest a folder
    python -m aiar.rag.ingest ./examples/docs --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)

# ~400 tokens at ~4 chars/token, with a small overlap so a fact split across a
# chunk boundary still survives in at least one chunk.
_CHUNK_TARGET_CHARS = 1600
_CHUNK_OVERLAP_CHARS = 200

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".text"}
_JSON_SUFFIXES = {".json"}
SUPPORTED_SUFFIXES = _TEXT_SUFFIXES | _JSON_SUFFIXES


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


def _chunk_text(source: str, title: str, text: str, category: str) -> List[Chunk]:
    """Split text into overlapping chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if not paragraphs:
        flat = text.strip()
        paragraphs = [flat] if flat else []
    chunks: List[Chunk] = []
    current = ""
    prev_tail = ""
    for para in paragraphs:
        if len(current) + len(para) > _CHUNK_TARGET_CHARS and current:
            chunks.append(Chunk(
                source=source, title=title, chunk_index=len(chunks),
                text=(prev_tail + current).strip(), category=category,
            ))
            prev_tail = current[-_CHUNK_OVERLAP_CHARS:]
            current = ""
        current += ("\n\n" if current else "") + para
    if current.strip():
        chunks.append(Chunk(
            source=source, title=title, chunk_index=len(chunks),
            text=(prev_tail + current).strip(), category=category,
        ))
    return chunks


def ingest_file(path: Path, *, category: str = "general") -> List[Chunk]:
    """Read one document and return its chunks. Returns [] on any error."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("ingest: could not read %s: %s", path, exc)
        return []
    if path.suffix.lower() in _JSON_SUFFIXES:
        text = _json_to_text(raw)
    else:
        text = raw
    return _chunk_text(str(path), path.stem, text, category)


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
    parser.add_argument("folder", help="Path to a folder of .txt/.md/.json documents")
    parser.add_argument("--category", default="general",
                        help="Free-form metadata tag for the chunks (default: general)")
    parser.add_argument("--instance", default=None,
                        help="RAG instance to ingest into (default: the active "
                             "instance, i.e. RAG_INSTANCE or 'default'). A new "
                             "named instance is created on first ingest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be ingested without writing to the store")
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
    chunks = ingest_folder(folder, category=args.category)
    if not chunks:
        print("No chunks produced (no supported documents found).", file=sys.stderr)
        return 1
    if args.dry_run:
        _print_dry_run(chunks)
        return 0
    from aiar.rag import store
    store.init()
    instance = args.instance or store.active_instance()
    if instance == store.NO_RAG:
        print(f"error: cannot ingest into the No-RAG instance ({store.NO_RAG!r}); "
              f"pick a real instance with --instance or RAG_INSTANCE",
              file=sys.stderr)
        return 2
    written = store.add(chunks, instance=instance)
    print(f"Wrote {written} new chunks ({len(chunks)} candidates) "
          f"to instance {instance!r}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
