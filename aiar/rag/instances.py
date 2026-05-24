"""Named RAG instance registry + descriptor.

A RAG *instance* is a named, isolated corpus: its own ChromaDB collection, its
own derived BM25/title indexes, and its own grounding corrections. The registry
(a ``registry.json`` at ``<base>/knowledge/registry.json``) is the authoritative
list of which instances exist, their config, and published status. ``store``
self-heals it from ``client.list_collections()`` so a hand-created ``rag_*``
collection is never invisible.

AIAR is **born instance-aware**: there is NO migration and NO legacy alias. The
``default`` instance is created on first init; its collection name honours the
existing ``AIAR_CORPUS`` env so any pre-existing AIAR corpus is preserved.

Domain-agnostic by construction — instance names are free-form slugs, and the
optional per-instance ``query_rewrite`` prompts are operator config, never
hard-coded domain text.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_INSTANCE = "default"
_COLLECTION_PREFIX = "rag_"
_REGISTRY_FILE = "registry.json"
_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(name: str) -> str:
    """Canonicalise a free-form name into a slug usable as an instance id."""
    s = (name or "").strip().lower()
    s = _SLUG_RE.sub("-", s).strip("-")
    return s or "instance"


@dataclass
class InstanceDescriptor:
    """One registry entry. Domain-agnostic: ``query_rewrite`` prompts are
    optional operator config, defaulting to None (use the generic built-ins)."""

    name: str
    display_name: str
    collection: str
    embedding_model: str = "all-MiniLM-L6-v2"
    status: str = "draft"  # draft | published
    query_rewrite: Optional[Dict[str, str]] = None  # {rewrite_system, hyde_system}
    rerank_model: Optional[str] = None
    created_at: str = field(default_factory=_iso_now)
    published_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InstanceDescriptor":
        return cls(
            name=str(d.get("name", "")),
            display_name=str(d.get("display_name", "") or d.get("name", "")),
            collection=str(d.get("collection", "")),
            embedding_model=str(d.get("embedding_model", "all-MiniLM-L6-v2")),
            status=str(d.get("status", "draft")),
            query_rewrite=d.get("query_rewrite") or None,
            rerank_model=d.get("rerank_model") or None,
            created_at=str(d.get("created_at", "") or _iso_now()),
            published_at=d.get("published_at") or None,
        )


def collection_name(instance: str, *, default_collection: str) -> str:
    """Map an instance slug to its ChromaDB collection name.

    ``default`` maps to ``default_collection`` (honours ``AIAR_CORPUS``); every
    other instance is ``rag_<slug>``.
    """
    if instance == DEFAULT_INSTANCE:
        return default_collection
    return f"{_COLLECTION_PREFIX}{instance}"


def instance_from_collection(name: str) -> Optional[str]:
    """Inverse of ``collection_name`` for self-heal: a ``rag_*`` collection maps
    back to its instance slug. Returns None for non-prefixed names (the default
    collection is registered explicitly, not discovered this way)."""
    if name.startswith(_COLLECTION_PREFIX):
        slug = name[len(_COLLECTION_PREFIX):]
        return slug or None
    return None


class Registry:
    """JSON-backed map of ``{name: InstanceDescriptor}`` at
    ``<base>/knowledge/registry.json``. ``base`` is pinnable for tests."""

    def __init__(self, base: Path, *, default_collection: str) -> None:
        self._path = base / "knowledge" / _REGISTRY_FILE
        self._default_collection = default_collection
        self._lock = threading.Lock()
        self._entries: Dict[str, InstanceDescriptor] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for name, raw in data.items():
                    if isinstance(raw, dict):
                        self._entries[name] = InstanceDescriptor.from_dict(raw)
        except (OSError, ValueError):
            self._entries = {}
        self._ensure_default()

    def _ensure_default(self) -> None:
        if DEFAULT_INSTANCE not in self._entries:
            self._entries[DEFAULT_INSTANCE] = InstanceDescriptor(
                name=DEFAULT_INSTANCE,
                display_name="Example RAG",
                collection=self._default_collection,
                status="published",
                published_at=_iso_now(),
            )
            self._save()
            return
        desc = self._entries[DEFAULT_INSTANCE]
        if desc.display_name in ("", "Default"):
            desc.display_name = "Example RAG"
            self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: d.to_dict() for name, d in self._entries.items()}
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)

    def get(self, name: str) -> Optional[InstanceDescriptor]:
        return self._entries.get(name)

    def exists(self, name: str) -> bool:
        return name in self._entries

    def names(self) -> List[str]:
        return list(self._entries.keys())

    def all(self) -> List[InstanceDescriptor]:
        return list(self._entries.values())

    def create(self, name: str, *, display_name: Optional[str] = None,
               query_rewrite: Optional[Dict[str, str]] = None,
               rerank_model: Optional[str] = None) -> InstanceDescriptor:
        """Register a new draft instance. Idempotent — returns the existing
        descriptor if already present (does not clobber config)."""
        slug = slugify(name)
        with self._lock:
            if slug in self._entries:
                return self._entries[slug]
            desc = InstanceDescriptor(
                name=slug,
                display_name=display_name or name,
                collection=collection_name(
                    slug, default_collection=self._default_collection),
                query_rewrite=query_rewrite,
                rerank_model=rerank_model,
            )
            self._entries[slug] = desc
            self._save()
            return desc

    def publish(self, name: str) -> InstanceDescriptor:
        with self._lock:
            desc = self._entries.get(name)
            if desc is None:
                raise ValueError(f"unknown instance: {name!r}")
            desc.status = "published"
            desc.published_at = _iso_now()
            self._save()
            return desc

    def backfill(self, name: str) -> InstanceDescriptor:
        """Register a collection discovered on disk that has no registry entry
        (self-heal). Surfaces it as a draft with default config."""
        with self._lock:
            if name in self._entries:
                return self._entries[name]
            desc = InstanceDescriptor(
                name=name,
                display_name=name,
                collection=collection_name(
                    name, default_collection=self._default_collection),
                status="draft",
            )
            self._entries[name] = desc
            self._save()
            return desc
