"""Optional FastAPI service exposing the AIAR prompt harness over HTTP.

    POST /eval/prompt {"prompt": "..."}            -> answer + verdict
    POST /eval/prompt?rag=false                    -> blind the answerer (A/B)
    POST /eval/prompt?think=true                   -> include visible reasoning
    POST /eval/prompt?reground=true                -> prepend prior corrections
    POST /services/prompt                          -> external-service prompt
    POST /reground {"prompt": "...", "score": 4,   -> record a correction into
                    "correction": "...", "reason",
                    "instance": "docs"}              the grounding store
    GET  /services/meta                            -> service discovery / runtime state
    GET  /capabilities                             -> capability manifest (gate features)
    GET  /calls/{call_id}                          -> redacted trace for a prior call
    GET  /healthz                                  -> readiness snapshot

Run with:  uvicorn aiar.harness.service:app --port 8765
(Requires the optional ``service`` extra: ``pip install fastapi uvicorn``.)

This is intentionally thin — all logic lives in ``aiar.harness.pipeline`` and
``aiar.grounding`` so the same behaviour is available from the CLI, the A/B
runner, and the watcher GUI without going through HTTP.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "aiar.harness.service requires fastapi + uvicorn: "
        "pip install fastapi uvicorn\n(original error: %s)" % exc
    )

import hashlib
import os
import socket

from aiar import __version__ as _AIAR_VERSION
from aiar.contracts.capabilities import serialize_capabilities
from aiar.contracts.retrieve import RETRIEVE_SCHEMA_VERSION
from aiar.llm import OllamaError, active_model, healthcheck, list_models
from aiar.observability import observer
from aiar.rag import store
from aiar.harness.pipeline import ANSWER_SYSTEM_PROMPT, active_system_prompt, answer_prompt
from aiar.harness.admin_routes import router as admin_router
from aiar.harness import auth
from aiar.grounding import store as grounding_store
from aiar.eval.schemas import Verdict

app = FastAPI(title="AIAR harness", version="0.2.4")

# Authenticated remote ingest + instance-management routes (loopback + token).
# Existing query/eval routes below are unchanged.
app.include_router(admin_router)


def _remote_ingest_mounted() -> bool:
    """True iff the authenticated ingest/instance routes are mounted on this app.

    Derived from the mounted routes — not a hand-set flag — so it can never lie
    about what this process serves. Handles both how ``include_router`` is
    represented across FastAPI versions: a single ``_IncludedRouter`` entry
    (identity match) or the sub-routes flattened into ``app.routes`` (path match).
    """
    return any(
        getattr(r, "original_router", None) is admin_router
        or getattr(r, "path", "").startswith("/instances")
        for r in app.routes
    )


def _remote_ingest_enabled() -> bool:
    """Whether remote ingest is actually *usable* right now: the routes are
    mounted AND a token is configured.

    Remote clients decide whether ingest is available from this marker (not
    from ``store_ready``/``embedder_ready``). A box with the routes mounted but no
    ``AIAR_SERVICE_TOKEN`` set rejects every write with 503 (fail-closed), so it
    must report ``false`` here rather than look capable and fail on first push.
    """
    return _remote_ingest_mounted() and auth._configured_token() is not None


def _pure_retrieve_mounted() -> bool:
    """True iff the authenticated pure-retrieve route is mounted on this app."""
    retrieve_path = "/instances/{instance}/retrieve"
    router_has_route = any(
        getattr(r, "path", "") == retrieve_path
        for r in getattr(admin_router, "routes", [])
    )
    return any(
        getattr(r, "path", "") == retrieve_path
        or (getattr(r, "original_router", None) is admin_router and router_has_route)
        for r in app.routes
    )


def _pure_retrieve_enabled() -> bool:
    """Whether pure retrieve is usable right now: mounted and token configured."""
    return _pure_retrieve_mounted() and auth._configured_token() is not None


def _backend_id() -> str:
    """Stable identifier for "which AIAR is this app talking to". Deterministic
    from the DB path + hostname, so it is stable across restarts without needing
    to persist anything. Carries no secret material."""
    raw = f"{store._db_path()}|{socket.gethostname()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _trace_debug_enabled() -> bool:
    """Local debug flag: when set, ``GET /calls/{id}`` includes prompt/corpus
    bytes. Off by default — traces are redacted."""
    return (os.environ.get("AIAR_TRACE_DEBUG", "").strip().lower()
            in ("1", "true", "yes", "on"))


@app.on_event("startup")
def _startup() -> None:
    store.init()


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)
    context: Optional[str] = None
    instance: Optional[str] = None


class ServicePromptRequest(BaseModel):
    service_name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    context: Optional[str] = None
    instance: Optional[str] = None
    model: Optional[str] = None
    system: Optional[str] = None
    rag: bool = True
    judge: bool = False
    think: bool = False
    reground: bool = False
    sources: bool = False
    top_k: Optional[int] = Field(default=None, ge=1, le=200)
    metadata: Optional[Dict[str, Any]] = None


@app.post("/eval/prompt")
def eval_prompt(req: PromptRequest, rag: bool = True, think: bool = False,
                reground: bool = False) -> dict:
    try:
        return answer_prompt(
            req.prompt, rag=rag, judge=True, think=think,
            reground=True if reground else None, context=req.context or "",
            instance=req.instance,
        )
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail={"code": "ollama_error", "error": str(exc)})


@app.post("/services/prompt")
def service_prompt(req: ServicePromptRequest) -> dict:
    """Generic service-facing prompt endpoint."""
    try:
        result = answer_prompt(
            req.prompt,
            rag=req.rag,
            judge=req.judge,
            think=req.think,
            reground=True if req.reground else None,
            top_k=req.top_k,
            context=req.context or "",
            instance=req.instance,
            model=req.model,
            system=req.system,
            endpoint="/services/prompt",
            include_sources=req.sources,
        )
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail={"code": "ollama_error", "error": str(exc)})
    result["service_name"] = req.service_name
    result["service_metadata"] = req.metadata or {}
    return result


class RegroundRequest(BaseModel):
    prompt: str = Field(min_length=1)
    score: int = Field(ge=1, le=10)
    correction: str = ""
    reason: str = ""
    instance: Optional[str] = None


def _score_to_rating(score: int) -> str:
    if score >= 8:
        return "good"
    if score >= 4:
        return "partial"
    return "bad"


@app.post("/reground")
def reground(req: RegroundRequest) -> dict:
    """Record an evaluated (prompt, score, correction) into the grounding store
    so future answers for this prompt are corrected."""
    verdict = Verdict(rating=_score_to_rating(req.score),
                      reason=req.reason or req.correction,
                      failure_tags=[], confidence="high")
    # Keep the existing /reground response shape byte-identical (Correction echo:
    # rating/ts). The new product-safe ``record_grounding`` API is available for
    # callers that want answer/correction separation; this legacy endpoint stays
    # on the legacy writer (which persists data the new lookup reads back fine).
    rec = grounding_store.record(
        req.prompt, verdict, correction=req.correction, instance=req.instance)
    return {"ok": True, "recorded": rec.to_dict()}


@app.get("/healthz")
def healthz() -> dict:
    rag = store.health()
    ollama_ok = healthcheck()
    return {
        "ok": ollama_ok and rag.get("store_ready"),
        "ollama_reachable": ollama_ok,
        "remote_ingest": _remote_ingest_enabled(),
        "pure_retrieve": _pure_retrieve_enabled(),
        "retrieve_schema_version": RETRIEVE_SCHEMA_VERSION,
        "rag": rag,
    }


@app.get("/capabilities")
def capabilities() -> dict:
    """Capability manifest (``aiar.capabilities.v1``). Consumers gate UI
    affordances on this, never on the version string. ``features`` come from the
    same live predicates ``/healthz`` uses (mounted AND usable), so the manifest
    can't claim a capability this process can't serve."""
    return serialize_capabilities(
        aiar_version=_AIAR_VERSION,
        backend_id=_backend_id(),
        pure_retrieve=_pure_retrieve_enabled(),
        remote_ingest=_remote_ingest_enabled(),
        grounding_v1=True,
        semantic_grounding=False,  # A5, deferred
        judge_only=False,          # A5
        streaming=False,           # A5
        answer_sources=True,
        call_trace=True,
    )


@app.get("/calls/{call_id}")
def call_trace(call_id: str) -> dict:
    """Compact, redacted trace for a prior call. Best-effort and process-local:
    traces are date-partitioned JSONL with a FIFO cap, so old ``call_id``s age
    out (404). Prompt/corpus bytes are redacted unless ``AIAR_TRACE_DEBUG`` is
    set — token counts and ``call_id`` are never redacted."""
    event = observer.read_by_call_id(call_id)
    if event is None:
        raise HTTPException(status_code=404,
                            detail={"code": "unknown_call", "error": call_id})
    debug = _trace_debug_enabled()
    redacted = "[redacted: set AIAR_TRACE_DEBUG to reveal]"
    err = event.get("error")
    trace = {
        "call_id": event.get("call_id"),
        "timestamp": event.get("timestamp"),
        "endpoint": event.get("endpoint"),
        "model": event.get("model"),
        "think": event.get("think"),
        "done_reason": event.get("done_reason"),
        "prompt_tokens": event.get("prompt_tokens"),
        "completion_tokens": event.get("completion_tokens"),
        "latency_ms": event.get("latency_ms"),
        "error": err,
        "debug": debug,
    }
    for key in ("raw_prompt", "system_prompt", "user_prompt",
                "response_text", "thinking"):
        trace[key] = event.get(key) if debug else redacted
    return trace


@app.get("/services/meta")
def services_meta() -> dict:
    """External-service metadata snapshot.

    This is intentionally read-only and generic so sibling services can inspect
    AIAR's model/RAG surface without depending on the watcher GUI endpoints.
    """
    rag = store.health()
    instances = store.list_instances()
    ollama_ok = healthcheck()
    system_text = active_system_prompt()
    return {
        "ok": ollama_ok and rag.get("store_ready"),
        "ollama_reachable": ollama_ok,
        "active_model": active_model(),
        "available_models": list_models(show_all=True),
        "rag": {
            **rag,
            "instances": instances,
        },
        "system_prompt": {
            "has_override": system_text != ANSWER_SYSTEM_PROMPT,
            "label": "Custom override"
            if system_text != ANSWER_SYSTEM_PROMPT
            else "Built-in default",
        },
    }
