"""Runtime overrides for retrieval-quality features.

Each feature (hybrid retrieval, cross-encoder reranker, query rewrite/HyDE,
grounding reinjection, top-k, fetch-k) is normally env-driven and read per call.
This module adds a process-global **override** layer so the watcher GUI can toggle
a feature live, with no restart. The effective value resolves:

    runtime override  ->  environment variable  ->  built-in default

Overrides are in-process only — a restart reverts to the env/defaults (same
semantics as the live model / RAG-instance / system-prompt switches). ``reset()``
clears them (this is the "reset/turn-off" that stands in for "delete", since a
framework is a capability, not stored data). Nothing here imports the rest of the
RAG stack, so ``retriever`` / ``query_rewrite`` / grounding can route their flag
reads through this without an import cycle.
"""
from __future__ import annotations

import os
from typing import Any, Dict

# key -> (env var, default, type)
_SPEC: "Dict[str, tuple]" = {
    "hybrid": ("RAG_HYBRID_ENABLED", False, "bool"),
    "rerank": ("RAG_RERANK_ENABLED", False, "bool"),
    "fetch_k": ("RAG_FETCH_K", 20, "int"),
    "top_k": ("RAG_TOP_K", 3, "int"),
    "rewrite_mode": ("RAG_QUERY_REWRITE_MODE", "off", "mode"),
    "grounding_reinjection": ("GROUNDING_REINJECTION_ENABLED", False, "bool"),
}
_MODES = ("off", "rewrite", "hyde")
_TRUE = ("1", "true", "yes", "on")
_INT_MIN, _INT_MAX = 1, 200
_RERANK_MODEL_ENV = "RAG_RERANK_MODEL"
_DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_overrides: Dict[str, Any] = {}


def keys() -> "list[str]":
    return list(_SPEC.keys())


def _coerce(key: str, value: Any) -> Any:
    """Validate + coerce a value for ``key``; raise ValueError if invalid."""
    _, _, t = _SPEC[key]
    if t == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in _TRUE:
            return True
        if isinstance(value, str) and value.strip().lower() in ("0", "false", "no", "off", ""):
            return False
        raise ValueError(f"{key} must be a boolean")
    if t == "int":
        try:
            iv = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer")
        if not (_INT_MIN <= iv <= _INT_MAX):
            raise ValueError(f"{key} must be between {_INT_MIN} and {_INT_MAX}")
        return iv
    if t == "mode":
        v = str(value).strip().lower()
        if v not in _MODES:
            raise ValueError(f"rewrite_mode must be one of {_MODES}")
        return v
    raise ValueError(f"unknown type for {key}")  # pragma: no cover


def _env_value(key: str) -> "tuple[Any, str]":
    """Return (value, source) from env or default for ``key``."""
    env, default, t = _SPEC[key]
    raw = os.environ.get(env)
    if raw is None or raw == "":
        return default, "default"
    try:
        if t == "bool":
            return raw.strip().lower() in _TRUE, "env"
        if t == "int":
            return int(raw), "env"
        if t == "mode":
            v = raw.strip().lower()
            return (v if v in _MODES else default), "env"
    except (TypeError, ValueError):
        return default, "default"
    return default, "default"  # pragma: no cover


def get(key: str) -> Any:
    """Effective value: override -> env -> default."""
    if key in _overrides:
        return _overrides[key]
    return _env_value(key)[0]


def source(key: str) -> str:
    """Where the effective value comes from: 'override' | 'env' | 'default'."""
    if key in _overrides:
        return "override"
    return _env_value(key)[1]


def rerank_model() -> str:
    return os.environ.get(_RERANK_MODEL_ENV) or _DEFAULT_RERANK_MODEL


def effective() -> Dict[str, Any]:
    """The full effective config (what the next answer will use)."""
    cfg = {k: get(k) for k in _SPEC}
    cfg["rerank_model"] = rerank_model()
    return cfg


def sources() -> Dict[str, str]:
    return {k: source(k) for k in _SPEC}


def defaults() -> Dict[str, Any]:
    return {k: _SPEC[k][1] for k in _SPEC}


def set_override(key: str, value: Any) -> Any:
    """Set a live override for ``key``. Raises ValueError on bad key/value."""
    if key not in _SPEC:
        raise ValueError(f"unknown retrieval setting: {key!r}")
    _overrides[key] = _coerce(key, value)
    return _overrides[key]


def reset() -> None:
    """Clear all runtime overrides (revert to env/defaults)."""
    _overrides.clear()


# tests pin/clear process-global state through this name
reset_for_testing = reset
