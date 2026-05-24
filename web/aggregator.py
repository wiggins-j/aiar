"""Business logic for the AIAR watcher GUI.

Generic over any corpus — it reads the local observer JSONL logs (every LLM call
the harness made) and drives the evaluate/reground flow. No domain logic.

Flow primitives exposed to the HTTP layer:
    - simulate_prompt(prompt, rag, think)   : run a prompt through the harness
    - recent_activity(limit)                : tail recent LLM calls
    - activity_detail(call_id)              : one call's full prompt/response
    - enqueue(call_id)                      : mark a call for evaluation
    - evaluation_queue()                    : list pending evaluations
    - submit_verdict(call_id, score, ...)   : score it AND reground (record the
                                              correction into the grounding store)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiar.observability import observer
from aiar.eval.schemas import Verdict
from aiar.grounding import store as grounding_store
from .config import Config


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# JSONL helpers
# --------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    items.append(parsed)
    except OSError:
        return []
    return items


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _summary(prompt: str, limit: int = 80) -> str:
    s = " ".join((prompt or "").split())
    return (s[:limit] + "…") if len(s) > limit else (s or "(empty prompt)")


# --------------------------------------------------------------------------
# Simulate (run a prompt through the harness)
# --------------------------------------------------------------------------

def simulate_prompt(prompt: str, *, rag: bool = True, think: bool = False,
                    reground: bool = False, instance: Optional[str] = None,
                    model: Optional[str] = None,
                    system: Optional[str] = None) -> Dict[str, Any]:
    """Run ``prompt`` through the harness and return the answer + verdict.

    The harness logs the call to the observer, so the returned ``call_id`` is
    immediately markable for evaluation. ``instance``/``model``/``system`` are
    passed straight through as per-request overrides (None -> the active
    instance/model/system prompt).
    """
    from aiar.harness import answer_prompt
    result = answer_prompt(prompt, rag=rag, judge=True, think=think,
                           reground=True if reground else None,
                           instance=instance, model=model, system=system)
    result["prompt"] = prompt
    return result


# --------------------------------------------------------------------------
# Settings (in-process) — model switch, RAG instance switch, system prompt.
#
# AIAR's web GUI imports the brain in-process (no separate proxy hop), so these
# call aiar.llm / aiar.rag.store / the harness module directly. The /api/...
# JSON contract is what the Settings page consumes; only the server-side
# fulfillment lives here.
# --------------------------------------------------------------------------

def get_models() -> Dict[str, Any]:
    """List installed models + the active/default model. Never raises; an
    unreachable Ollama degrades to {models: [], ollama_reachable: false}."""
    from aiar.llm import ollama_client
    models = ollama_client.list_models()
    active = ollama_client.active_model()
    reachable = bool(models)
    return {
        "active": active,
        "default": ollama_client.default_model(),
        "models": [
            {**m, "active": m.get("name") == active} for m in models
        ],
        "source": "ollama/api/tags",
        "ollama_reachable": reachable,
    }


def set_active_model(model: str) -> Dict[str, Any]:
    """Switch the process-active model. 422 if not installed."""
    from aiar.llm import ollama_client
    model = (model or "").strip()
    if not model:
        return {"ok": False, "status": 400, "error": "missing_model"}
    previous = ollama_client.active_model()
    try:
        ollama_client.set_active_model(model)
    except ValueError as exc:
        return {"ok": False, "status": 422,
                "error": "model_not_installed", "data": str(exc)}
    return {"ok": True, "status": 200,
            "data": {"active": model, "previous": previous}}


def get_rag_instances() -> Dict[str, Any]:
    """List RAG instances + the active instance. Includes the first-class
    'No RAG' (``none``) selectable so the page renders it without special-casing."""
    from aiar.rag import store
    if not store.is_ready():
        try:
            store.init()
        except Exception:
            pass
    instances = store.list_instances()
    active = store.active_instance()
    return {
        "active": active,
        "instances": instances,
        "no_rag_option": {"name": "none", "display_name": "No RAG"},
    }


def set_active_rag(name: str) -> Dict[str, Any]:
    """Switch the process-active RAG instance. ``none`` is selectable (No RAG)."""
    from aiar.rag import store
    name = (name or "").strip()
    if not name:
        return {"ok": False, "status": 400, "error": "missing_name"}
    if name == "none":
        # "No RAG" is a resolution-boundary concept; we record it as active so
        # instance-less reads skip retrieval. The store itself never gets a
        # ``none`` handle — set_active stores the sentinel for the read path.
        store.set_active_none()
        return {"ok": True, "status": 200, "data": {"active": "none"}}
    try:
        store.set_active(name)
    except ValueError as exc:
        return {"ok": False, "status": 422,
                "error": "unknown_instance", "data": str(exc)}
    return {"ok": True, "status": 200, "data": {"active": name}}


def get_system_prompt() -> Dict[str, Any]:
    """Return the active harness system prompt (+ whether it's an override or
    the built-in default)."""
    from aiar.harness import pipeline
    active = pipeline.active_system_prompt()
    source = "active" if active != pipeline.ANSWER_SYSTEM_PROMPT else "default"
    return {"text": active, "default": pipeline.ANSWER_SYSTEM_PROMPT,
            "source": source}


def set_system_prompt(text: str) -> Dict[str, Any]:
    """Set (or, with empty text, reset) the active harness system prompt. This
    governs ONLY the harness/answer path (guardrail)."""
    from aiar.harness import pipeline
    pipeline.set_system_prompt(text)
    active = pipeline.active_system_prompt()
    source = "active" if active != pipeline.ANSWER_SYSTEM_PROMPT else "default"
    return {"ok": True, "status": 200,
            "data": {"text": active, "source": source}}


# --------------------------------------------------------------------------
# Recent activity (tail the observer log)
# --------------------------------------------------------------------------

def recent_activity(config: Config, limit: int = 25) -> Dict[str, Any]:
    events = observer.read_recent(limit)
    verdicts = _verdicts_by_call(config)
    queued = _queued_by_call(config)
    items: List[Dict[str, Any]] = []
    for ev in reversed(events):  # newest first
        cid = str(ev.get("call_id") or "")
        items.append({
            "call_id": cid,
            "timestamp": ev.get("timestamp"),
            "endpoint": ev.get("endpoint"),
            "model": ev.get("model"),
            "summary": _summary(ev.get("user_prompt") or ""),
            "latency_ms": ev.get("latency_ms"),
            "status": _status(cid, queued, verdicts),
        })
    return {"items": items, "count": len(items), "generated_at": iso_now()}


def activity_detail(config: Config, call_id: str) -> Dict[str, Any]:
    ev = observer.read_by_call_id(call_id)
    if not ev:
        return {"found": False, "call_id": call_id}
    verdicts = _verdicts_by_call(config)
    queued = _queued_by_call(config)
    return {
        "found": True,
        "call_id": call_id,
        "timestamp": ev.get("timestamp"),
        "endpoint": ev.get("endpoint"),
        "model": ev.get("model"),
        "prompt": ev.get("user_prompt"),
        "response": ev.get("response_text"),
        "thinking": ev.get("thinking"),
        "error": ev.get("error"),
        "status": _status(call_id, queued, verdicts),
    }


# --------------------------------------------------------------------------
# Evaluation queue
# --------------------------------------------------------------------------

def _queued_by_call(config: Config) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in _read_jsonl(config.queue_file):
        cid = str(rec.get("call_id") or "").strip()
        if cid:
            out.setdefault(cid, rec)
    return out


def _verdicts_by_call(config: Config) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in _read_jsonl(config.verdicts_file):
        cid = str(rec.get("call_id") or "").strip()
        if cid:
            out[cid] = rec  # latest wins
    return out


def _status(call_id: str, queued: Dict[str, Any], verdicts: Dict[str, Any]) -> Dict[str, Any]:
    if call_id in verdicts:
        v = verdicts[call_id]
        return {"status": "complete", "label": "Evaluated", "score": v.get("score"),
                "rating": v.get("rating")}
    if call_id in queued:
        return {"status": "pending", "label": "Pending", "score": None, "rating": None}
    return {"status": "none", "label": "Not queued", "score": None, "rating": None}


def enqueue(config: Config, call_id: str) -> Dict[str, Any]:
    """Mark a logged call for evaluation. Idempotent."""
    ev = observer.read_by_call_id(call_id)
    if not ev:
        return {"ok": False, "status": 404, "error": "call_not_found"}
    if not str(ev.get("response_text") or "").strip():
        return {"ok": False, "status": 422, "error": "not_rateable"}
    if call_id not in _queued_by_call(config) and call_id not in _verdicts_by_call(config):
        _append_jsonl(config.queue_file, {
            "call_id": call_id,
            "queued_at": iso_now(),
            "endpoint": ev.get("endpoint"),
            "model": ev.get("model"),
            "prompt": ev.get("user_prompt"),
            "response": ev.get("response_text"),
        })
    return {"ok": True, "status": 200, "data": {"call_id": call_id, "status": "pending"}}


def evaluation_queue(config: Config) -> Dict[str, Any]:
    verdicts = _verdicts_by_call(config)
    items: List[Dict[str, Any]] = []
    for rec in _read_jsonl(config.queue_file):
        cid = str(rec.get("call_id") or "").strip()
        if not cid or cid in verdicts:
            continue
        items.append({
            "call_id": cid,
            "queued_at": rec.get("queued_at"),
            "endpoint": rec.get("endpoint"),
            "summary": _summary(rec.get("prompt") or ""),
            "prompt": rec.get("prompt"),
            "response": rec.get("response"),
        })
    return {"items": items, "count": len(items),
            "reason_threshold": config.reason_threshold, "generated_at": iso_now()}


def _score_to_rating(score: int) -> str:
    if score >= 8:
        return "good"
    if score >= 4:
        return "partial"
    return "bad"


def submit_verdict(config: Config, call_id: str, score: int,
                   correction: str = "") -> Dict[str, Any]:
    """Record a 1-10 verdict for a queued call AND reground.

    Scores at/below the reason threshold require a ``correction`` (what the
    answer should have been). The verdict is appended to the verdicts log, and
    the correction is recorded into the grounding store keyed by the original
    prompt's signature — so the next answer for that prompt is corrected. This
    IS the Reground action.
    """
    ev = observer.read_by_call_id(call_id)
    if not ev:
        return {"ok": False, "status": 404, "error": "call_not_found"}
    correction = (correction or "").strip()
    if score <= config.reason_threshold and not correction:
        return {"ok": False, "status": 422, "error": "correction_required"}

    prompt = ev.get("user_prompt") or ""
    rating = _score_to_rating(score)
    verdict = Verdict(rating=rating, reason=correction, failure_tags=[], confidence="high")

    # 1. Persist the human verdict.
    _append_jsonl(config.verdicts_file, {
        "call_id": call_id,
        "rated_at": iso_now(),
        "score": score,
        "rating": rating,
        "correction": correction,
        "prompt": prompt,
        "response": ev.get("response_text"),
    })
    # 2. Reground: feed the correction back into the grounding store.
    rec = grounding_store.record(prompt, verdict, correction=correction)
    return {"ok": True, "status": 200,
            "data": {"call_id": call_id, "rating": rating, "score": score,
                     "regrounded": True, "correction_ts": rec.ts}}
