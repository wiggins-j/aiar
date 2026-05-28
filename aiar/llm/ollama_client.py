"""Ollama HTTP client. The only file in AIAR that talks to Ollama.

Model-agnostic: ``OLLAMA_MODEL`` selects ANY Qwen model you have pulled
(``ollama pull qwen2.5:7b``, ``ollama pull qwen2.5:3b``, etc.). The default
fallback is a real, small, pullable tag; verify current tags at
https://ollama.com/library/qwen (specs: https://huggingface.co/Qwen).
Swapping models is a config change, not a code edit.

Keeps the think/capture handling and native-thinking-field support from the
upstream client: Qwen "thinking" models can emit a dedicated top-level
``thinking`` field (or inline ``<think>...</think>`` tags); we strip those out
of the answer and surface the reasoning separately via ``capture``.
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional

import requests

from aiar.observability import observer
from aiar.observability.observer import _THINK_BLOCK_RE
from aiar import runtime_state


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
# Default is a placeholder — point this at ANY Qwen model you have pulled.
_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")  # env boot seed
_active_model = _DEFAULT_MODEL                                  # runtime-mutable
MODEL = _DEFAULT_MODEL                                          # back-compat alias
DEFAULT_TIMEOUT_SECONDS = 30
_LENGTH_RETRY_DONE_REASONS = {"length", "max_tokens"}
_LENGTH_RETRY_MAX_ATTEMPTS = 4
_LENGTH_RETRY_MIN_NUM_PREDICT = 1200
_LENGTH_RETRY_MAX_NUM_PREDICT = 4800
_LENGTH_RETRY_TIMEOUT_FLOOR_S = 90
_LENGTH_RETRY_TIMEOUT_CAP_S = 300


class OllamaError(Exception):
    """Wraps any Ollama call failure (including timeout)."""


# --------------------------------------------------------------------------
# Active-model layer — runtime model switch, no restart. Resolution order:
#   explicit model= arg  ->  process-active  ->  OLLAMA_MODEL env  ->  default.
# --------------------------------------------------------------------------

def active_model() -> str:
    """The model every call path resolves to when no explicit ``model=`` is
    passed. Runtime-mutable via :func:`set_active_model` (no restart)."""
    return str(runtime_state.get("active_model") or _active_model)


def default_model() -> str:
    """The env boot seed (``OLLAMA_MODEL``) — what a restart reverts to."""
    return _DEFAULT_MODEL


def set_active_model(name: str) -> None:
    """Validate ``name`` against the installed models, then flip the
    process-global active model. Raises ``ValueError`` if not installed.

    Validates against the UNFILTERED installed set (``show_all=True``) so a
    model-agnostic caller can switch to any installed model — not just those
    matching ``MODEL_LIST_PREFIXES`` (which only governs the dropdown)."""
    installed = {m["name"] for m in list_models(show_all=True)}
    if name not in installed:
        raise ValueError(
            f"model not installed: {name!r} (installed: {sorted(installed)})")
    global _active_model
    _active_model = name
    runtime_state.set_value("active_model", name)


def _model_prefixes() -> "list[str]":
    raw = os.environ.get("MODEL_LIST_PREFIXES", "qwen")
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def list_models(*, show_all: bool = False) -> "list[dict]":
    """Query Ollama ``/api/tags`` for installed models.

    Returns ``[{name, size_bytes, family}]``. By default filters to the family
    prefixes in ``MODEL_LIST_PREFIXES`` (default ``qwen``); an empty env or
    ``show_all=True`` returns everything. Returns ``[]`` if Ollama is
    unreachable — never raises (graceful degradation for the Settings page)."""
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        r = requests.get(f"{base}/api/tags", timeout=2)
        if r.status_code != 200:
            return []
        body = r.json()
    except (requests.RequestException, ValueError):
        return []
    prefixes = _model_prefixes()
    out: "list[dict]" = []
    for m in (body.get("models") or []):
        name = m.get("name") or m.get("model")
        if not name:
            continue
        family = (m.get("details") or {}).get("family") or name.split(":", 1)[0]
        if not show_all and prefixes:
            hay = f"{name} {family}".lower()
            if not any(p in hay for p in prefixes):
                continue
        out.append({
            "name": name,
            "size_bytes": m.get("size"),
            "family": family,
        })
    return out


def reset_for_testing() -> None:
    """Reset the active model to the current ``OLLAMA_MODEL`` env seed. Tests
    call this so the process-global active model never leaks across tests."""
    global _DEFAULT_MODEL, _active_model, MODEL
    _DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    _active_model = _DEFAULT_MODEL
    MODEL = _DEFAULT_MODEL
    runtime_state.reset_for_testing()


def _strip_thinking(text: str) -> str:
    """Remove thinking-mode artefacts before JSON parsing / display.

    Qwen "thinking" models can leak ``<think>...</think>`` blocks or
    ``Thinking...done thinking.`` markers even with ``think: false``; strip
    them defensively so the returned answer is clean prose / JSON.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    text = re.sub(r"Thinking\.\.\..*?\.\.\.done thinking\.", "", text, flags=re.DOTALL)
    text = re.sub(r"Thinking Process:.*?(?=\{)", "", text, flags=re.DOTALL)
    return text.strip()


def _should_retry_on_length(done_reason: object, *, retry_on_length: bool) -> bool:
    if not retry_on_length:
        return False
    reason = str(done_reason or "").strip().lower()
    return reason in _LENGTH_RETRY_DONE_REASONS


def _expanded_num_predict(raw: object) -> int:
    try:
        current = int(raw)
    except (TypeError, ValueError):
        current = 0
    if current <= 0:
        return _LENGTH_RETRY_MIN_NUM_PREDICT
    return min(
        max(current * 2, _LENGTH_RETRY_MIN_NUM_PREDICT),
        _LENGTH_RETRY_MAX_NUM_PREDICT,
    )


def call_ollama(
    system_prompt: str,
    user_prompt: str,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    model: Optional[str] = None,
    options_override: Optional[dict] = None,
    format: Optional[str] = "json",
    think: bool = False,
    capture: Optional[dict] = None,
    retry_on_length: bool = False,
) -> "tuple[str, int]":
    """Call Ollama. Returns ``(response_text, latency_ms)``.

    Raises :class:`OllamaError` on any failure (including timeout).

    ``model`` is a sentinel: ``None`` resolves to the process-active model
    (:func:`active_model`) at call time, so every caller that passes no
    ``model=`` automatically follows a runtime switch. Pass an explicit model
    to override the active model for one call (per-request A/B).

    ``options_override`` shallow-merges into the default Ollama ``options``.
    ``format`` is the Ollama structured-output mode — defaults to ``"json"``;
    pass ``None`` for free-text prose answers (forcing JSON on a prose
    question degrades the answer).

    ``think`` controls a Qwen thinking model's reasoning mode. Defaults to
    ``False``. The returned answer is always the post-strip text; the
    ``<think>...</think>`` content is surfaced via ``capture`` (below), never
    appended to the answer.

    ``capture`` is an optional caller-provided dict; when supplied it is
    populated with ``thinking`` (raw CoT or ``None``), ``response_text``,
    ``latency_ms``, ``prompt_tokens`` and ``completion_tokens``.

    ``retry_on_length`` retries with progressively larger token budgets when
    Ollama reports the completion stopped because it hit the token budget. The
    retry path doubles ``num_predict`` (with a sensible floor/cap) and raises
    the timeout floor so long answers are not silently logged mid-sentence.
    """
    model = model if model is not None else active_model()
    options = {
        "temperature": 0.2,
        "top_p": 0.9,
        "num_predict": 400,
        "num_ctx": 4096,
    }
    if options_override:
        options.update(options_override)

    result = {
        "response_text": None,
        "thinking": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "done_reason": None,
        "error": None,
    }
    emitted_options = dict(options)
    raised: Optional[BaseException] = None
    start = time.monotonic()
    try:
        attempt_options = dict(options)
        attempt_timeout = timeout_s
        max_attempts = _LENGTH_RETRY_MAX_ATTEMPTS if retry_on_length else 1
        for attempt in range(max_attempts):
            payload = {
                "model": model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "think": think,
                "options": attempt_options,
            }
            if format is not None:
                payload["format"] = format

            try:
                r = requests.post(OLLAMA_URL, json=payload, timeout=attempt_timeout)
                r.raise_for_status()
            except requests.Timeout:
                raised = OllamaError(f"Ollama timeout after {attempt_timeout}s")
                raise raised
            except requests.RequestException as e:
                raised = OllamaError(f"Ollama request failed: {e}")
                raise raised

            try:
                body = r.json()
            except ValueError as e:
                raised = OllamaError(f"Ollama returned non-JSON: {e}")
                raise raised

            if "response" not in body:
                raised = OllamaError(f"Ollama response missing 'response' field: {body}")
                raise raised

            raw_response = body["response"]
            # Modern Ollama returns chain-of-thought in a dedicated ``thinking``
            # field when think=true; fall back to inline-<think> extraction.
            native_thinking = body.get("thinking")
            thinking_content = native_thinking or observer.extract_thinking(raw_response)
            clean_response = _strip_thinking(raw_response)
            result["response_text"] = clean_response
            result["thinking"] = thinking_content
            result["prompt_tokens"] = body.get("prompt_eval_count")
            result["completion_tokens"] = body.get("eval_count")
            result["done_reason"] = body.get("done_reason")
            emitted_options = dict(attempt_options)

            if _should_retry_on_length(result["done_reason"], retry_on_length=retry_on_length):
                next_num_predict = _expanded_num_predict(attempt_options.get("num_predict"))
                current_num_predict = attempt_options.get("num_predict")
                if str(next_num_predict) != str(current_num_predict) and attempt + 1 < max_attempts:
                    attempt_options = dict(attempt_options)
                    attempt_options["num_predict"] = next_num_predict
                    attempt_timeout = min(
                        max(
                            float(attempt_timeout) * 2.0,
                            float(timeout_s),
                            float(_LENGTH_RETRY_TIMEOUT_FLOOR_S),
                        ),
                        float(_LENGTH_RETRY_TIMEOUT_CAP_S),
                    )
                    continue

            latency_ms = int((time.monotonic() - start) * 1000)
            if capture is not None:
                capture["thinking"] = thinking_content
                capture["response_text"] = clean_response
                capture["latency_ms"] = latency_ms
                capture["prompt_tokens"] = result["prompt_tokens"]
                capture["completion_tokens"] = result["completion_tokens"]
                capture["done_reason"] = result["done_reason"]
            return clean_response, latency_ms
    finally:
        latency_ms = int((time.monotonic() - start) * 1000)
        if raised is not None:
            result["error"] = {
                "class": type(raised).__name__,
                "message": str(raised),
            }
        call_id = observer.emit_call(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            options=emitted_options,
            format=format,
            think=think,
            response_text=result["response_text"],
            thinking=result["thinking"],
            done_reason=result["done_reason"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            latency_ms=latency_ms,
            error=result["error"],
        )
        if capture is not None:
            capture["call_id"] = call_id


def healthcheck() -> bool:
    """Quick check that Ollama is reachable. Returns True/False, never raises."""
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        r = requests.get(f"{base}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False
