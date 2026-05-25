"""Pre-retrieval query rewrite / HyDE.

The text we embed is whatever the question literally says; the source documents
phrase it differently. The terse query's vector can land far from the corpus
prose, so the right chunk doesn't retrieve even though it exists. Transforming
the query *before* retrieval closes that vocabulary gap:

  * rewrite — normalise/expand the question into corpus-like phrasing.
  * hyde    — draft a hypothetical answer and retrieve against THAT (a fake
              answer is lexically closer to the real chunk than the bare
              question is).

Recall-side; complements rerank/hybrid (which act on the candidate set after
retrieval). Reuses the single Ollama client; one extra call max per retrieval,
and it degrades to the raw query on timeout/error so retrieval never blocks.

Mode is set via ``RAG_QUERY_REWRITE_MODE`` (``off`` | ``rewrite`` | ``hyde``),
default ``off``.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_REWRITE_SYS = (
    "You rewrite a user's question into a concise search query for a document "
    "corpus. Use precise terminology and make implicit constraints explicit. "
    "Output ONLY the query, no preamble."
)
_HYDE_SYS = (
    "You are a knowledge base. Write a single concise paragraph that answers "
    "the question the way an authoritative document would. Output ONLY the "
    "paragraph."
)


def _mode() -> str:
    from aiar.rag import settings as rag_settings  # local import: no import cycle
    return str(rag_settings.get("rewrite_mode") or "off").strip().lower()


def _timeout_s() -> float:
    raw = os.environ.get("RAG_QUERY_REWRITE_TIMEOUT_MS")
    try:
        ms = int(raw) if raw is not None else 1500
    except ValueError:
        ms = 1500
    return max(0.5, ms / 1000.0)


def _instance_prompts(instance: "str | None") -> "tuple[str, str]":
    """Resolve (rewrite_system, hyde_system) for ``instance``.

    Domain-generic by default: uses the built-in generic prompts unless the
    instance descriptor carries operator-set per-instance prompts. Never
    hard-codes domain text.
    """
    rewrite_sys, hyde_sys = _REWRITE_SYS, _HYDE_SYS
    if instance is None:
        return rewrite_sys, hyde_sys
    try:
        from aiar.rag import store
        desc = store.descriptor(instance)
    except Exception:
        desc = None
    qr = getattr(desc, "query_rewrite", None) if desc is not None else None
    if isinstance(qr, dict):
        rewrite_sys = qr.get("rewrite_system") or rewrite_sys
        hyde_sys = qr.get("hyde_system") or hyde_sys
    return rewrite_sys, hyde_sys


def transform(query: str, *, instance: "str | None" = None) -> str:
    """Return the (possibly rewritten) query to retrieve on. Mode ``off`` ->
    returns ``query`` unchanged with no LLM call. Any failure or degenerate
    output (empty / echoes the prompt) -> falls back to the raw query.

    ``instance`` selects per-instance rewrite/HyDE system prompts from the
    descriptor when set (else the generic built-ins)."""
    mode = _mode()
    if mode not in ("rewrite", "hyde"):
        return query
    rewrite_sys, hyde_sys = _instance_prompts(instance)
    try:
        out = (hyde(query, system=hyde_sys) if mode == "hyde"
               else rewrite(query, system=rewrite_sys))
    except Exception as exc:
        logger.debug("query_rewrite[%s] failed (%s) — using raw query", mode, exc)
        return query
    out = (out or "").strip()
    if not out or out.lower() == (query or "").strip().lower():
        return query
    logger.info("query_rewrite[%s]: %r -> %r", mode, query, out)
    return out


def rewrite(query: str, *, system: "str | None" = None) -> str:
    return _call(system or _REWRITE_SYS, query)


def hyde(query: str, *, system: "str | None" = None) -> str:
    return _call(system or _HYDE_SYS, f"Write a one-paragraph answer to: {query}")


def _call(system: str, user: str) -> str:
    from aiar.llm import call_ollama  # local import to avoid import cycle

    text, _latency = call_ollama(
        system, user,
        timeout_s=_timeout_s(),
        format=None,  # prose, not JSON
        options_override={"num_predict": 120},
    )
    return text
