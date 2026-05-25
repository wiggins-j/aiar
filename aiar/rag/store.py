"""ChromaDB-backed vector store for AIAR document chunks — instance-aware.

AIAR is **born instance-aware**: one ChromaDB persist dir holds N collections,
one per named RAG instance. A single shared embedder serves them all; only the
collection (and the derived indexes) are per-instance.

Resolution order for any read with no explicit ``instance=``:
    explicit arg  ->  process-active (``set_active``)  ->  ``RAG_INSTANCE`` env
    ->  ``default``.

The ``default`` instance's collection name honours the existing ``AIAR_CORPUS``
env so any pre-existing AIAR corpus is preserved with NO data move (born-aware,
not migrated). Newly-created instances use the ``rag_<name>`` collection prefix.

Write isolation: ``add`` *requires* an explicit ``instance=`` — there is no
"add to whatever's global" path, so ingest can never silently pollute the active
instance.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from aiar.rag import instances
from aiar.rag.ingest import Chunk

logger = logging.getLogger(__name__)

# AIAR never phones home. Disable ChromaDB's anonymized telemetry before it is
# imported (the env var is read at import time) and silence its telemetry logger
# as a belt-and-suspenders — some chromadb versions emit a noisy capture() error
# from a background posthog client regardless of client Settings.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "chromadb.telemetry.product.posthog.Posthog")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


def _db_path() -> str:
    return os.environ.get("AIAR_DB_PATH", str(Path.home() / ".aiar" / "knowledge"))


def _embedding_model() -> str:
    return os.environ.get("AIAR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def _default_collection() -> str:
    # Honour AIAR_CORPUS so an existing single-corpus deploy is preserved as the
    # ``default`` instance with no data move.
    return os.environ.get("AIAR_CORPUS", "aiar")


def _base_dir() -> Path:
    """Registry base dir. Defaults to the parent of the DB path so registry.json
    sits beside the ChromaDB data (``<base>/knowledge/registry.json``)."""
    if _BASE_OVERRIDE is not None:
        return _BASE_OVERRIDE
    return Path(_db_path()).parent


# --- process-global state ---------------------------------------------------
_client = None
_embedder = None
_available = False
_registry: "Optional[instances.Registry]" = None
_collections: Dict[str, object] = {}   # instance name -> chromadb collection
_active: Optional[str] = None           # process-active instance; None until init
_BASE_OVERRIDE: Optional[Path] = None   # test pin for the registry base


@dataclass
class RetrievedChunk:
    """One retrieval hit: chunk text plus provenance and similarity.

    ``score`` is cosine similarity (1 - ChromaDB cosine distance); higher is
    better.
    """
    id: str
    text: str
    score: float
    metadata: dict


def init() -> None:
    """Initialise ChromaDB + the shared embedder + the instance registry.

    Degrades gracefully: if chromadb / sentence-transformers are not installed,
    the store is simply unavailable and every query returns ``[]`` rather than
    raising — so the harness still runs (just without RAG).
    """
    global _client, _embedder, _available, _registry, _collections, _active
    try:
        import chromadb
        from chromadb.config import Settings as _ChromaSettings
        from sentence_transformers import SentenceTransformer

        db_path = _db_path()
        Path(db_path).mkdir(parents=True, exist_ok=True)
        # AIAR never phones home: disable ChromaDB's anonymized telemetry (it also
        # emits a noisy capture() error on some versions). Privacy + clean output.
        _client = chromadb.PersistentClient(
            path=db_path, settings=_ChromaSettings(anonymized_telemetry=False))
        _embedder = SentenceTransformer(_embedding_model())
        _collections = {}
        _registry = instances.Registry(
            _base_dir(), default_collection=_default_collection())
        _self_heal()
        _active = _resolve_boot_active()
        _available = True
        logger.info("store: ready (instances=%s, active=%s)",
                    _registry.names(), _active)
    except Exception as exc:
        logger.error("store: init failed — running without RAG: %s", exc)
        _available = False


def _self_heal() -> None:
    """Back-fill any ``rag_*`` collection that has no registry entry."""
    if _client is None or _registry is None:
        return
    try:
        for col in _client.list_collections():
            name = getattr(col, "name", None)
            if not name:
                continue
            inst = instances.instance_from_collection(name)
            if inst and not _registry.exists(inst):
                _registry.backfill(inst)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("store: self-heal failed: %s", exc)


def _resolve_boot_active() -> str:
    env = os.environ.get("RAG_INSTANCE")
    if env and _registry is not None and _registry.exists(env):
        return env
    return instances.DEFAULT_INSTANCE


def _resolve(instance: Optional[str]) -> str:
    """arg -> process-active -> RAG_INSTANCE env -> default."""
    if instance:
        return instance
    if _active:
        return _active
    env = os.environ.get("RAG_INSTANCE")
    if env:
        return env
    return instances.DEFAULT_INSTANCE


# "No RAG" sentinel — when the active instance is this, instance-less reads
# resolve to no collection (retrieval is skipped at the boundary).
NO_RAG = "none"


def _handle(instance: Optional[str]):
    """Return (or lazily open + cache) the ChromaDB collection for ``instance``.

    Auto-registers the descriptor if missing so a per-request instance that was
    never explicitly created still resolves (it opens its own collection).
    Returns None for the ``none`` (No RAG) sentinel — there is no collection.
    """
    if not _available or _client is None or _registry is None:
        return None
    name = _resolve(instance)
    if name == NO_RAG:
        return None
    if name in _collections:
        return _collections[name]
    desc = _registry.get(name)
    if desc is None:
        desc = _registry.create(name)
    col = _client.get_or_create_collection(
        name=desc.collection, metadata={"hnsw:space": "cosine"})
    _collections[name] = col
    return col


# --- lifecycle / instance management ---------------------------------------

def create_instance(name: str, *, display_name: Optional[str] = None,
                    query_rewrite: Optional[dict] = None,
                    rerank_model: Optional[str] = None) -> str:
    """Register a draft instance + create its (empty) collection. Idempotent.
    Returns the resolved slug."""
    if _registry is None:
        init()
    if _registry is None:
        raise RuntimeError("store unavailable")
    desc = _registry.create(name, display_name=display_name,
                            query_rewrite=query_rewrite, rerank_model=rerank_model)
    _handle(desc.name)  # open the collection so chunk_count is queryable
    return desc.name


def publish_instance(name: str) -> None:
    if _registry is None:
        raise RuntimeError("store unavailable")
    _registry.publish(name)


def delete_instance(name: str) -> dict:
    """Delete a RAG instance: drop its ChromaDB collection, registry entry, and
    cached collection handle + BM25 index. The ``default`` instance and the
    ``none`` sentinel cannot be deleted. If the deleted instance was active, the
    active resets to ``default``. Returns ``{"deleted": name, "active": ...}``.
    """
    global _active
    if _registry is None:
        init()
    if not _available or _client is None or _registry is None:
        raise RuntimeError("store unavailable")
    name = (name or "").strip()
    if not name or name == NO_RAG:
        raise ValueError(f"cannot delete: {name!r}")
    if name == instances.DEFAULT_INSTANCE:
        raise ValueError("cannot delete the default instance")
    desc = _registry.get(name)
    if desc is None:
        raise ValueError(f"unknown instance: {name!r}")
    try:
        _client.delete_collection(desc.collection)
    except Exception as exc:  # already gone / never created — proceed to deregister
        logger.debug("store: delete_collection(%s) failed: %s", desc.collection, exc)
    _collections.pop(name, None)
    _registry.delete(name)
    try:
        from aiar.rag import lexical
        lexical.invalidate(instance=name)
    except Exception:  # pragma: no cover - defensive
        pass
    # Purge this instance's grounding corrections (its own on-disk subdir) so a
    # delete leaves no orphaned per-RAG state behind. The shared infrastructure
    # (registry file, store, flat/global corrections) is untouched.
    try:
        import shutil
        from aiar.grounding.store import GroundingStore
        gdir = GroundingStore(instance=name).root
        if gdir.is_dir():
            shutil.rmtree(gdir, ignore_errors=True)
    except Exception:  # pragma: no cover - defensive
        pass
    if _active == name:
        _active = instances.DEFAULT_INSTANCE
    return {"deleted": name, "active": active_instance()}


def set_active(name: str) -> None:
    """Flip the process-global active instance (no restart). Validates the
    instance is registered. Accepts the ``none`` sentinel (No RAG)."""
    global _active
    if name == NO_RAG:
        _active = NO_RAG
        return
    if _registry is None or not _registry.exists(name):
        raise ValueError(f"unknown instance: {name!r}")
    _active = name


def set_active_none() -> None:
    """Set the active instance to the No-RAG sentinel — instance-less reads then
    skip retrieval (the answerer is blinded)."""
    global _active
    _active = NO_RAG


def active_instance() -> str:
    return _resolve(None)


def descriptor(instance: Optional[str] = None):
    """Return the InstanceDescriptor for the resolved instance, or None."""
    if _registry is None:
        return None
    return _registry.get(_resolve(instance))


def list_instances() -> List[dict]:
    """Return ``[{name, display_name, status, chunk_count, active}]`` for every
    registered instance (self-healed). The Settings dropdown source."""
    if _registry is None:
        return []
    active = active_instance()
    out: List[dict] = []
    for desc in _registry.all():
        out.append({
            "name": desc.name,
            "display_name": desc.display_name,
            "status": desc.status,
            "chunk_count": chunk_count(instance=desc.name) or 0,
            "active": desc.name == active,
        })
    return out


# --- reads / writes (instance-aware) ---------------------------------------

# ChromaDB rejects a single get()/add() larger than its internal cap (~5461,
# from the SQLite variable limit), so batch large ingests under it.
_MAX_ADD_BATCH = 5000


def add(chunks: List[Chunk], *, instance: str) -> int:
    """Embed and store chunks into ``instance``'s collection. Skip duplicates.
    Returns count added. ``instance`` is REQUIRED (write isolation). Large
    ingests are written in batches so a corpus over ChromaDB's per-call cap
    (~5461 items) does not fail."""
    col = _handle(instance)
    if col is None or not chunks:
        return 0
    try:
        ids = [_chunk_id(c) for c in chunks]
        existing: set = set()
        for i in range(0, len(ids), _MAX_ADD_BATCH):
            got = col.get(ids=ids[i:i + _MAX_ADD_BATCH])
            existing.update(got.get("ids") or [])
        new = [(c, i) for c, i in zip(chunks, ids) if i not in existing]
        if not new:
            return 0
        new_chunks, new_ids = zip(*new)
        embeddings = _embedder.encode([c.text for c in new_chunks]).tolist()
        documents = [c.text for c in new_chunks]
        metadatas = [{"source": c.source, "title": c.title,
                      "index": c.chunk_index, "category": c.category}
                     for c in new_chunks]
        for i in range(0, len(new_chunks), _MAX_ADD_BATCH):
            sl = slice(i, i + _MAX_ADD_BATCH)
            col.add(ids=list(new_ids[sl]), embeddings=embeddings[sl],
                    documents=documents[sl], metadatas=metadatas[sl])
        return len(new_chunks)
    except Exception as exc:
        logger.error("store: add failed: %s", exc)
        return 0


def query_scored(
    text: str,
    n_results: int = 3,
    where: Optional[dict] = None,
    *,
    instance: Optional[str] = None,
) -> List[RetrievedChunk]:
    """Embed query, return top-N hits from ``instance``'s collection.

    ``where`` is an optional ChromaDB metadata filter. Returns [] if the store
    is unavailable, the collection is empty, or the query raises.
    """
    col = _handle(instance)
    if col is None:
        return []
    try:
        count = col.count()
        if count == 0:
            return []
        embedding = _embedder.encode([text]).tolist()
        results = col.query(
            query_embeddings=embedding,
            n_results=min(n_results, count),
            where=where or None,
        )
        docs = (results.get("documents") or [[]])[0]
        if not docs:
            return []
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        hits: List[RetrievedChunk] = []
        for i, doc in enumerate(docs):
            dist = distances[i] if i < len(distances) else None
            hits.append(RetrievedChunk(
                id=ids[i] if i < len(ids) else "",
                text=doc,
                score=(1.0 - dist) if dist is not None else 0.0,
                metadata=metadatas[i] if i < len(metadatas) else {},
            ))
        return hits
    except Exception as exc:
        logger.error("store: query_scored failed: %s", exc)
        return []


def query(text: str, n_results: int = 3, *,
          instance: Optional[str] = None) -> List[str]:
    """Embed query, return top-N chunk texts. Thin wrapper over query_scored."""
    return [c.text for c in query_scored(text, n_results, instance=instance)]


def is_ready() -> bool:
    """True iff the store initialised successfully."""
    return _available


def chunk_count(*, instance: Optional[str] = None) -> Optional[int]:
    """Number of chunks in ``instance``'s collection, or None if unavailable."""
    col = _handle(instance)
    if col is None:
        return None
    try:
        return col.count()
    except Exception:
        return None


def all_documents(*, instance: Optional[str] = None
                  ) -> "tuple[List[str], List[str]]":
    """Return (ids, documents) for ``instance``'s whole collection.

    Used to build the lexical BM25 index from the single source of truth (the
    ChromaDB docs) — no second copy of the corpus on disk.
    """
    col = _handle(instance)
    if col is None:
        return [], []
    try:
        got = col.get(include=["documents"])
        return list(got.get("ids") or []), list(got.get("documents") or [])
    except Exception as exc:
        logger.error("store: all_documents failed: %s", exc)
        return [], []


def get_by_ids(ids: List[str], *,
               instance: Optional[str] = None) -> List[RetrievedChunk]:
    """Materialise chunks by id from ``instance``'s collection, preserving input
    order. (Coupling 5b: scoped to the instance — no global-collection read.)

    ``score`` is left 0.0 — callers that need a score (e.g. RRF fusion) assign
    their own. Missing ids are skipped.
    """
    col = _handle(instance)
    if col is None or not ids:
        return []
    try:
        got = col.get(ids=ids, include=["documents", "metadatas"])
        got_ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        by_id = {
            cid: (docs[i] if i < len(docs) else "", metas[i] if i < len(metas) else {})
            for i, cid in enumerate(got_ids)
        }
        out: List[RetrievedChunk] = []
        for cid in ids:
            if cid in by_id:
                text, meta = by_id[cid]
                out.append(RetrievedChunk(id=cid, text=text, score=0.0, metadata=meta or {}))
        return out
    except Exception as exc:
        logger.error("store: get_by_ids failed: %s", exc)
        return []


def reset(*, instance: Optional[str] = None) -> None:
    """Delete and recreate ``instance``'s collection. Used by tests / re-ingest."""
    if not _available or _client is None or _registry is None:
        return
    name = _resolve(instance)
    desc = _registry.get(name) or _registry.create(name)
    try:
        _client.delete_collection(desc.collection)
    except Exception:
        pass
    _collections.pop(name, None)
    try:
        _collections[name] = _client.get_or_create_collection(
            name=desc.collection, metadata={"hnsw:space": "cosine"})
    except Exception as exc:
        logger.error("store: reset failed: %s", exc)


def reset_for_testing(*, base: Optional[Path] = None) -> None:
    """Fully reset the process-global store state and (optionally) re-init
    against a pinned registry ``base``. Tests call this so the process-global
    store never pollutes across tests."""
    global _client, _embedder, _available, _registry, _collections, _active
    global _BASE_OVERRIDE
    _client = None
    _embedder = None
    _available = False
    _registry = None
    _collections = {}
    _active = None
    _BASE_OVERRIDE = Path(base) if base is not None else None
    if base is not None:
        init()


def _chunk_id(chunk: Chunk) -> str:
    src_hash = hashlib.sha256(chunk.source.encode()).hexdigest()[:8]
    return f"{src_hash}-{chunk.chunk_index}"
