"""JSONL observer for every LLM call.

The hook is placed at the single chokepoint (``aiar.llm.call_ollama``).
Per-call context (endpoint/handler name, request_id) flows in via a
``ContextVar`` that callers set before invoking the LLM.

Failures here MUST never break an LLM call: every entry point catches
``Exception`` and logs at WARNING.

Log location is ``AIAR_LOG_DIR`` (default ``~/.aiar/logs/llm``). The watcher
GUI tails these ``calls-YYYY-MM-DD.jsonl`` files.
"""
from __future__ import annotations

import json
import logging
import os
import re as _re
import shutil
import time
import uuid
from collections import deque
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Source of truth for ``<think>...</think>`` block detection. Both
# ``extract_thinking`` here and ``_strip_thinking`` in the ollama client
# reference this compiled regex so they can never drift apart.
_THINK_BLOCK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)

LLM_CALL_CONTEXT: ContextVar[Optional[dict]] = ContextVar("LLM_CALL_CONTEXT", default=None)

_OBSERVER_OFF_SENTINEL = ".observer_off"
_MAX_BYTES_DEFAULT = 1 * 1024 ** 3  # 1 GB FIFO cap per active log file
_DEFAULT_LOG_DIR = "~/.aiar/logs/llm"


def _resolve_log_dir() -> Path:
    """Resolve the log directory at call time (env override honoured per call)."""
    return Path(os.environ.get("AIAR_LOG_DIR", _DEFAULT_LOG_DIR)).expanduser()


def log_dir() -> Path:
    """Public read-only accessor for the active observer log directory."""
    return _resolve_log_dir()


def _truthy(raw: Optional[str], default: bool) -> bool:
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def observer_paused() -> bool:
    """True when logging is OFF via ``AIAR_OBSERVER_ENABLED=0`` or a
    ``.observer_off`` sentinel in the log dir."""
    if not _truthy(os.environ.get("AIAR_OBSERVER_ENABLED"), True):
        return True
    try:
        return (_resolve_log_dir() / _OBSERVER_OFF_SENTINEL).exists()
    except Exception:
        return False


def set_logging(enabled: bool) -> bool:
    """Toggle logging by creating/removing the ``.observer_off`` sentinel."""
    d = _resolve_log_dir()
    d.mkdir(parents=True, exist_ok=True)
    sentinel = d / _OBSERVER_OFF_SENTINEL
    if enabled:
        sentinel.unlink(missing_ok=True)
    else:
        sentinel.touch()
    return enabled


def _max_bytes() -> int:
    try:
        return int(os.environ.get("AIAR_OBSERVER_MAX_BYTES", _MAX_BYTES_DEFAULT))
    except (TypeError, ValueError):
        return _MAX_BYTES_DEFAULT


def _enforce_fifo_cap(path: Path) -> None:
    """FIFO: if the active log exceeds the cap, drop the OLDEST lines, keeping
    the most recent ~half (line-aligned). Best-effort — never raises."""
    try:
        cap = _max_bytes()
        if cap <= 0 or not path.exists():
            return
        size = path.stat().st_size
        if size <= cap:
            return
        keep = cap // 2
        tmp = path.with_suffix(".jsonl.fifo.tmp")
        with open(path, "rb") as src:
            src.seek(max(0, size - keep))
            src.readline()  # discard the partial first line
            with open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
        os.replace(tmp, path)
    except Exception as exc:
        _log.warning("observer FIFO trim failed: %s", exc)


def _persist(payload: dict) -> None:
    """Append one JSONL line to today's calls log. Never raises."""
    if observer_paused():
        return
    d = _resolve_log_dir()
    d.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = d / f"calls-{today}.jsonl"
    _enforce_fifo_cap(path)
    line = json.dumps(payload, default=str, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_context(endpoint: str, request_id: Optional[str] = None, **extra: Any) -> Token:
    """Set per-call observation context. The token MUST be passed to
    :func:`clear_context` in a ``try/finally`` by the caller."""
    ctx = {"endpoint": endpoint, "request_id": request_id}
    ctx.update(extra)
    return LLM_CALL_CONTEXT.set(ctx)


def clear_context(token: Token) -> None:
    try:
        LLM_CALL_CONTEXT.reset(token)
    except Exception as exc:
        _log.warning("observer.clear_context failed: %s", exc)


def extract_thinking(raw_response: str) -> Optional[str]:
    """Pull ``<think>...</think>`` content out of a raw Ollama response, or
    ``None`` if none. Multiple blocks are joined with ``\\n---\\n``."""
    if not raw_response:
        return None
    matches = _THINK_BLOCK_RE.findall(raw_response)
    if not matches:
        return None
    inner = []
    for m in matches:
        body = m[len("<think>"):-len("</think>")].strip()
        if body:
            inner.append(body)
    return "\n---\n".join(inner) if inner else None


def emit_call(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    options: dict,
    format: Optional[str],
    think: bool,
    response_text: Optional[str],
    thinking: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    latency_ms: int,
    error: Optional[dict],
) -> Optional[str]:
    """Append one JSONL event describing this Ollama call. Best-effort.

    Returns the emitted ``call_id`` when persistence succeeds, else ``None``.
    """
    try:
        ctx = LLM_CALL_CONTEXT.get() or {}
        call_id = str(uuid.uuid4())
        event = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_id": call_id,
            "endpoint": ctx.get("endpoint") or "unknown",
            "request_id": ctx.get("request_id"),
            "raw_prompt": ctx.get("raw_prompt"),
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "options": options,
            "format": format,
            "think": think,
            "response_text": response_text,
            "thinking": thinking,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "error": error,
        }
        _persist(event)
        return call_id
    except Exception as exc:
        _log.warning("observer.emit_call failed: %s", exc)
        return None


def read_recent(limit: int = 100) -> "list[dict[str, Any]]":
    """Return the last ``limit`` parsed call events, oldest first."""
    if limit <= 0:
        return []
    files = sorted(_resolve_log_dir().glob("calls-*.jsonl"))
    if not files:
        return []
    remaining = limit
    chunks: "list[list[dict[str, Any]]]" = []
    for path in reversed(files):
        raw_lines: "deque[str]" = deque(maxlen=remaining)
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        raw_lines.append(line)
        except OSError:
            continue
        parsed: "list[dict[str, Any]]" = []
        for line in raw_lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                parsed.append(event)
        if not parsed:
            continue
        chunks.append(parsed)
        remaining -= len(parsed)
        if remaining <= 0:
            break
    recent: "list[dict[str, Any]]" = []
    for chunk in reversed(chunks):
        recent.extend(chunk)
    return recent[-limit:]


def read_by_call_id(call_id: str) -> "Optional[dict[str, Any]]":
    """Return the most recent observer event for ``call_id``, or ``None``."""
    target = str(call_id or "").strip()
    if not target:
        return None
    files = sorted(_resolve_log_dir().glob("calls-*.jsonl"))
    for path in reversed(files):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and str(event.get("call_id") or "").strip() == target:
                return event
    return None


def clear() -> int:
    """Delete every ``calls-*.jsonl`` activity log. Returns the number of call
    records removed. Best-effort — never raises. The active log is recreated on
    the next logged call."""
    removed = 0
    try:
        d = _resolve_log_dir()
        if not d.exists():
            return 0
        for p in d.glob("calls-*.jsonl"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    removed += sum(1 for line in f if line.strip())
            except OSError:
                pass
            try:
                p.unlink()
            except OSError as exc:
                _log.warning("observer.clear skipped %s: %s", p, exc)
    except Exception as exc:
        _log.warning("observer.clear failed: %s", exc)
    return removed


def prune_old(retention_days: int) -> int:
    """Delete ``calls-*.jsonl`` files older than ``retention_days``.
    ``<= 0`` means keep forever (no-op). Returns count deleted."""
    if retention_days <= 0:
        return 0
    deleted = 0
    try:
        d = _resolve_log_dir()
        if not d.exists():
            return 0
        cutoff = time.time() - retention_days * 86400
        for p in d.glob("calls-*.jsonl"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    deleted += 1
            except Exception as exc:
                _log.warning("observer.prune_old skipped %s: %s", p, exc)
    except Exception as exc:
        _log.warning("observer.prune_old failed: %s", exc)
    return deleted
