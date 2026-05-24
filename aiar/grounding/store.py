"""Grounding correction store — persisted verdicts + corrections.

The persistence half of the answer -> verdict -> grounded-correction ->
better-next-answer loop. It keys verdicts by a normalized prompt *signature*
and stores the corrected guidance so a later prompt-build can look matching
corrections up and reinject them (see ``aiar.grounding.reinject``).

Storage: one JSON file per signature at ``<base>/grounding/<hash>.json``, where
``<base>`` is configurable: the ``base`` arg wins, else ``GROUNDING_BASE_DIR``
env, else ``~/.aiar``. Tests pass a tmp ``base`` so the real dir is never
written. Each file holds the signature verbatim plus an append-only list of
correction records (newest last) — human-auditable.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

GROUNDING_BASE_DIR_ENV = "GROUNDING_BASE_DIR"
_DEFAULT_BASE_DIR = "~/.aiar"
_SUBDIR = "grounding"


def default_base_dir() -> Path:
    """Resolve the default grounding base path (env, else ``~/.aiar``)."""
    raw = os.environ.get(GROUNDING_BASE_DIR_ENV) or _DEFAULT_BASE_DIR
    return Path(raw).expanduser()


@dataclass
class Correction:
    """One recorded judgment + corrective guidance for a signature.

    - ``signature``: the (raw) prompt this correction is keyed to (verbatim).
    - ``rating`` / ``reason`` / ``failure_tags`` / ``confidence``: from the
      :class:`aiar.eval.schemas.Verdict` that produced this record.
    - ``correction``: the corrected guidance to reinject (free text). May be
      empty when the verdict was ``good``.
    - ``ts``: ISO-8601 UTC timestamp.
    """

    signature: str
    rating: str
    reason: str = ""
    correction: str = ""
    failure_tags: List[str] = field(default_factory=list)
    confidence: str = "medium"
    ts: str = ""

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "rating": self.rating,
            "reason": self.reason,
            "correction": self.correction,
            "failure_tags": list(self.failure_tags),
            "confidence": self.confidence,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Correction":
        return cls(
            signature=str(d.get("signature", "")),
            rating=str(d.get("rating", "")),
            reason=str(d.get("reason", "")),
            correction=str(d.get("correction", "")),
            failure_tags=list(d.get("failure_tags", []) or []),
            confidence=str(d.get("confidence", "medium")),
            ts=str(d.get("ts", "")),
        )


_WS_RE = re.compile(r"\s+")


def normalize_signature(signature: str) -> str:
    """Canonicalize a prompt into a stable matching key.

    v1 normalization is intentionally simple: lowercase, collapse whitespace,
    strip surrounding punctuation/space. Two prompts that differ only in
    case/spacing/trailing punctuation map to the same signature.
    """
    s = (signature or "").strip().lower()
    s = _WS_RE.sub(" ", s)
    return s.strip(" \t\n\r.?!")


def _signature_hash(normalized: str) -> str:
    return sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _iso_now() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class GroundingStore:
    """Synchronous JSON store of corrections keyed by normalized signature.

    Per-instance isolation: when ``instance`` is given, corrections live under
    ``<base>/grounding/<instance>/`` so a correction recorded against one RAG
    instance is never looked up for another. With no ``instance`` the legacy
    flat ``<base>/grounding/`` layout is used.
    """

    def __init__(self, base: Optional[Union[str, Path]] = None,
                 *, instance: Optional[str] = None) -> None:
        raw = Path(base).expanduser() if base is not None else default_base_dir()
        root = raw / _SUBDIR
        self._root = (root / instance) if instance else root
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, normalized: str) -> Path:
        return self._root / f"{_signature_hash(normalized)}.json"

    def _read_file(self, path: Path) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, ValueError) as exc:
            logger.debug("grounding: read %s failed: %s", path, exc)
        return {}

    def record(self, signature: str, verdict: object, correction: str = "") -> Correction:
        """Append a verdict + correction for ``signature``. Returns the record.

        ``verdict`` is duck-typed against :class:`aiar.eval.schemas.Verdict`
        (``rating`` / ``reason`` / ``failure_tags`` / ``confidence``) so this
        module need not import the eval package. ``correction`` is the corrective
        guidance to reinject later (free text).
        """
        normalized = normalize_signature(signature)
        rec = Correction(
            signature=signature,
            rating=str(getattr(verdict, "rating", "")),
            reason=str(getattr(verdict, "reason", "")),
            correction=correction or "",
            failure_tags=list(getattr(verdict, "failure_tags", []) or []),
            confidence=str(getattr(verdict, "confidence", "medium")),
            ts=_iso_now(),
        )
        with self._lock:
            self._root.mkdir(parents=True, exist_ok=True)
            path = self._path_for(normalized)
            data = self._read_file(path)
            entries = data.get("corrections")
            if not isinstance(entries, list):
                entries = []
            entries.append(rec.to_dict())
            payload = {"signature": signature, "normalized": normalized,
                       "corrections": entries}
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        return rec

    def lookup(self, signature: str) -> List[Correction]:
        """Return all recorded corrections for ``signature`` (newest last).

        Matching is by normalized signature. Returns [] when nothing recorded.
        """
        normalized = normalize_signature(signature)
        data = self._read_file(self._path_for(normalized))
        entries = data.get("corrections")
        if not isinstance(entries, list):
            return []
        return [Correction.from_dict(e) for e in entries if isinstance(e, dict)]


# --------------------------------------------------------------------------
# Module-level convenience seam
# --------------------------------------------------------------------------

_DEFAULT_STORE: Optional[GroundingStore] = None
_DEFAULT_STORE_LOCK = threading.Lock()


def _default_store() -> GroundingStore:
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = GroundingStore()
        return _DEFAULT_STORE


def reset_default_store_for_testing() -> None:
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        _DEFAULT_STORE = None


def record(signature: str, verdict: object, correction: str = "",
           *, base: Optional[Union[str, Path]] = None,
           instance: Optional[str] = None) -> Correction:
    """Record a correction. ``base`` pins the store root (tmp path in tests);
    ``instance`` scopes corrections to a RAG instance subdir."""
    if base is not None or instance is not None:
        store = GroundingStore(base, instance=instance)
    else:
        store = _default_store()
    return store.record(signature, verdict, correction)


def lookup(signature: str, *, base: Optional[Union[str, Path]] = None,
           instance: Optional[str] = None) -> List[Correction]:
    """Look up corrections. ``base`` pins the store root (tmp path in tests);
    ``instance`` scopes the lookup to a RAG instance subdir."""
    if base is not None or instance is not None:
        store = GroundingStore(base, instance=instance)
    else:
        store = _default_store()
    return store.lookup(signature)
