"""Optional FastAPI service exposing the AIAR prompt harness over HTTP.

    POST /eval/prompt {"prompt": "..."}            -> answer + verdict
    POST /eval/prompt?rag=false                    -> blind the answerer (A/B)
    POST /eval/prompt?think=true                   -> include visible reasoning
    POST /eval/prompt?reground=true                -> prepend prior corrections
    POST /reground {"prompt": "...", "score": 4,   -> record a correction into
                    "correction": "...", "reason",
                    "instance": "docs"}              the grounding store
    GET  /healthz                                  -> readiness snapshot

Run with:  uvicorn aiar.harness.service:app --port 8765
(Requires the optional ``service`` extra: ``pip install fastapi uvicorn``.)

This is intentionally thin — all logic lives in ``aiar.harness.pipeline`` and
``aiar.grounding`` so the same behaviour is available from the CLI, the A/B
runner, and the watcher GUI without going through HTTP.
"""
from __future__ import annotations

from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "aiar.harness.service requires fastapi + uvicorn: "
        "pip install fastapi uvicorn\n(original error: %s)" % exc
    )

from aiar.llm import OllamaError, healthcheck
from aiar.rag import store
from aiar.harness.pipeline import answer_prompt
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
