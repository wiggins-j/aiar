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

from aiar.llm import OllamaError, active_model, healthcheck, list_models
from aiar.rag import store
from aiar.harness.pipeline import ANSWER_SYSTEM_PROMPT, active_system_prompt, answer_prompt
from aiar.grounding import store as grounding_store
from aiar.eval.schemas import Verdict

app = FastAPI(title="AIAR harness", version="0.1.0")


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
        "rag": rag,
    }


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
