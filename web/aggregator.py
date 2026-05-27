"""Business logic for the AIAR watcher GUI.

Generic over any corpus — it reads the local observer JSONL logs (every LLM call
the harness made) and drives the evaluate/reground flow. No domain logic.

Flow primitives exposed to the HTTP layer:
    - simulate_prompt(prompt, rag, think)   : run a prompt through the harness
    - recent_activity(limit)                : tail recent LLM calls
    - activity_detail(call_id)              : one call's full prompt/response
    - enqueue(call_id)                      : mark a call for evaluation
    - evaluation_queue()                    : list pending evaluations
    - clear_evaluation_queue()              : drop the pending queue
    - submit_verdict(call_id, score, ...)   : score it AND reground (record the
                                              correction into the grounding store)
"""
from __future__ import annotations

import json
import os
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


def _preview(text: str, limit: int = 100) -> str:
    s = " ".join((text or "").split())
    return (s[:limit] + "…") if len(s) > limit else (s or "")


def _logged_prompt(event: Dict[str, Any]) -> str:
    """Prefer the original user-entered prompt when the observer captured it.

    Older log records only have the expanded harness prompt, so fall back to the
    raw LLM prompt payload for backward compatibility.
    """
    return str(event.get("raw_prompt") or event.get("user_prompt") or "")


def _rag_state(event: Dict[str, Any]) -> Optional[str]:
    """Best-effort RAG state for watcher rows.

    Only answer-path events carry ``raw_prompt``. For those, the presence of the
    labelled knowledge block means the answerer was grounded with retrieved
    context. Other call types (judge, rewrite, etc.) return None.
    """
    raw_prompt = str(event.get("raw_prompt") or "").strip()
    user_prompt = str(event.get("user_prompt") or "")
    if not raw_prompt:
        return None
    return "RAG ON" if "--- Knowledge" in user_prompt else "RAG OFF"


def _active_retrieval_features(result: Dict[str, Any]) -> List[str]:
    """Human-readable retrieval / grounding features active for one answer."""
    retrieval = result.get("retrieval") or {}
    labels: List[str] = []
    if retrieval.get("rag"):
        labels.append("Vector retrieval")
        if retrieval.get("hybrid"):
            labels.append("Hybrid")
        if retrieval.get("rerank"):
            labels.append("Rerank")
        mode = str(retrieval.get("rewrite_mode") or "off")
        if mode and mode != "off":
            labels.append("HyDE" if mode == "hyde" else "Rewrite")
        top_k = retrieval.get("top_k")
        if top_k is not None:
            labels.append(f"top-k {top_k}")
    else:
        labels.append("RAG off")
    if retrieval.get("grounding_reinjection"):
        labels.append("Grounding reinjection")
    if result.get("reground_applied"):
        labels.append("Correction block applied")
    return labels


def _grounding_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Compact grounding / retrieval facts for the Simulate page."""
    model_name = str(result.get("model") or "unknown")
    corpus_name = str(result.get("instance") or "default")
    chunk_count: "int | None" = None
    try:
        from aiar.rag import store
        if not store.is_ready():
            store.init()
        if corpus_name != "none":
            chunk_count = store.chunk_count(instance=corpus_name)
    except Exception:
        chunk_count = None
    return {
        "model_name": model_name,
        "corpus_name": corpus_name,
        "chunk_count": chunk_count,
        "system_prompt_name": _current_system_prompt_name(),
        "active_retrieval_features": _active_retrieval_features(result),
    }


def _current_system_prompt_name() -> str:
    """Human-friendly label for the currently active harness system prompt."""
    from aiar.harness import pipeline

    active = pipeline.active_system_prompt()
    if active == pipeline.ANSWER_SYSTEM_PROMPT:
        return "Built-in default"
    for preset in _read_presets():
        if preset.get("text", "") == active:
            return str(preset.get("name") or "Saved preset")
    return "Custom override"


def get_grounding_summary() -> Dict[str, Any]:
    """Current live grounding / retrieval setup for the Simulate page."""
    try:
        from aiar.llm import active_model
        model_name = active_model()
    except Exception:
        model_name = "unknown"

    try:
        from aiar.rag import store, settings as rag_settings
        if not store.is_ready():
            store.init()
        corpus_name = store.active_instance()
        chunk_count = None if corpus_name == "none" else store.chunk_count(instance=corpus_name)
        retrieval = rag_settings.effective()
    except Exception:
        corpus_name = "default"
        chunk_count = None
        retrieval = {}

    retrieval["rag"] = corpus_name != "none"
    return {
        "model_name": str(model_name),
        "corpus_name": str(corpus_name),
        "chunk_count": chunk_count,
        "system_prompt_name": _current_system_prompt_name(),
        "active_retrieval_features": _active_retrieval_features({
            "retrieval": retrieval,
            "reground_applied": False,
        }),
    }


# --------------------------------------------------------------------------
# Simulate (run a prompt through the harness)
# --------------------------------------------------------------------------

def simulate_prompt(prompt: str, *, rag: bool = True, think: bool = False,
                    reground: bool = False, judge: bool = True,
                    instance: Optional[str] = None,
                    model: Optional[str] = None,
                    system: Optional[str] = None) -> Dict[str, Any]:
    """Run ``prompt`` through the harness and return the answer + verdict.

    The harness logs the call to the observer, so the returned ``call_id`` is
    immediately markable for evaluation. ``instance``/``model``/``system`` are
    passed straight through as per-request overrides (None -> the active
    instance/model/system prompt). ``judge=False`` skips the LLM-as-judge so the
    result carries no verdict (the page renders 'judge: skipped').
    """
    from aiar.harness import answer_prompt
    result = answer_prompt(prompt, rag=rag, judge=judge, think=think,
                           reground=True if reground else None,
                           instance=instance, model=model, system=system)
    result["prompt"] = prompt
    result["grounding_summary"] = _grounding_summary(result)
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
    reachable = ollama_client.healthcheck()
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
    if active == "none":
        active_display_name = "No RAG"
    else:
        desc = store.descriptor(active)
        active_display_name = desc.display_name if desc else active
    return {
        "active": active,
        "active_display_name": active_display_name,
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
        return {"ok": True, "status": 200,
                "data": {"active": "none", "active_display_name": "No RAG"}}
    try:
        store.set_active(name)
    except ValueError as exc:
        return {"ok": False, "status": 422,
                "error": "unknown_instance", "data": str(exc)}
    desc = store.descriptor(name)
    return {"ok": True, "status": 200,
            "data": {"active": name,
                      "active_display_name": desc.display_name if desc else name}}


def delete_rag_instance(name: str) -> Dict[str, Any]:
    """Delete a RAG instance (its collection + registry entry). The ``default``
    instance and the ``none`` sentinel are not deletable. On success returns the
    now-active instance so the caller can refresh the dropdown."""
    from aiar.rag import store
    name = (name or "").strip()
    if not name:
        return {"ok": False, "status": 400, "error": "missing_name"}
    if name == "none":
        return {"ok": False, "status": 422, "error": "cannot_delete_none"}
    if not store.is_ready():
        try:
            store.init()
        except Exception:
            pass
    try:
        result = store.delete_instance(name)
    except ValueError as exc:
        return {"ok": False, "status": 422, "error": "delete_failed", "data": str(exc)}
    except RuntimeError as exc:
        return {"ok": False, "status": 503, "error": "store_unavailable", "data": str(exc)}
    active = result.get("active")
    if active == "none":
        active_display_name = "No RAG"
    else:
        desc = store.descriptor(active)
        active_display_name = desc.display_name if desc else active
    return {"ok": True, "status": 200,
            "data": {"deleted": result.get("deleted"), "active": active,
                     "active_display_name": active_display_name}}


def get_retrieval_settings() -> Dict[str, Any]:
    """Return the effective retrieval-feature config, where each value comes from
    (override/env/default), and the built-in defaults — for the Settings card."""
    from aiar.rag import settings as rag_settings
    return {
        "config": rag_settings.effective(),
        "sources": rag_settings.sources(),
        "defaults": rag_settings.defaults(),
    }


def set_retrieval_setting(key: str, value: Any) -> Dict[str, Any]:
    """Set one live retrieval-feature override (no restart). Validates key/value."""
    from aiar.rag import settings as rag_settings
    key = (key or "").strip()
    if key not in rag_settings.keys():
        return {"ok": False, "status": 400, "error": "unknown_setting", "data": key}
    try:
        rag_settings.set_override(key, value)
    except ValueError as exc:
        return {"ok": False, "status": 422, "error": "invalid_value", "data": str(exc)}
    return {"ok": True, "status": 200,
            "data": {"config": rag_settings.effective(),
                     "sources": rag_settings.sources(), "set": key}}


def reset_retrieval_settings() -> Dict[str, Any]:
    """Clear all live retrieval-feature overrides (revert to env/defaults)."""
    from aiar.rag import settings as rag_settings
    rag_settings.reset()
    return {"ok": True, "status": 200,
            "data": {"config": rag_settings.effective(),
                     "sources": rag_settings.sources()}}


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


# --- Named system-prompt presets (quick-save / load / delete) ----------------
#
# A small, persistent shelf of named system prompts so an operator can flip
# between a few guardrails without retyping. Stored as a JSON array at
# ``<base>/system-prompts.json`` (AIAR_BASE_DIR), independent of which prompt is
# currently active. Capped so the dropdown stays a quick-pick, not a database.

SYSTEM_PROMPT_PRESET_LIMIT = 5
SYSTEM_PROMPT_NAME_MAXLEN = 60


def _system_prompts_path() -> Path:
    base = Path(os.environ.get("AIAR_BASE_DIR", "~/.aiar")).expanduser()
    return Path(os.environ.get(
        "AIAR_SYSTEM_PROMPTS_FILE", str(base / "system-prompts.json"))).expanduser()


def _read_presets() -> List[Dict[str, str]]:
    path = _system_prompts_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: List[Dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and str(item.get("name") or "").strip():
                out.append({"name": str(item["name"]), "text": str(item.get("text") or "")})
    return out


def _write_presets(presets: List[Dict[str, str]]) -> None:
    path = _system_prompts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def list_system_prompts() -> Dict[str, Any]:
    """List saved system-prompt presets and the cap."""
    return {"presets": _read_presets(), "limit": SYSTEM_PROMPT_PRESET_LIMIT}


def save_system_prompt_preset(name: str, text: str) -> Dict[str, Any]:
    """Save (or overwrite by name) a named system-prompt preset. New names are
    rejected once the cap is reached; overwriting an existing name always works.
    Does NOT change the active system prompt — it just shelves the text."""
    name = (name or "").strip()
    text = text or ""
    if not name:
        return {"ok": False, "status": 400, "error": "missing_name"}
    if len(name) > SYSTEM_PROMPT_NAME_MAXLEN:
        return {"ok": False, "status": 422, "error": "name_too_long"}
    if not text.strip():
        return {"ok": False, "status": 422, "error": "empty_prompt"}
    presets = _read_presets()
    idx = next((i for i, p in enumerate(presets)
                if p["name"].lower() == name.lower()), None)
    if idx is None:
        if len(presets) >= SYSTEM_PROMPT_PRESET_LIMIT:
            return {"ok": False, "status": 422, "error": "preset_limit",
                    "data": f"at most {SYSTEM_PROMPT_PRESET_LIMIT} presets"}
        presets.append({"name": name, "text": text})
    else:
        presets[idx] = {"name": name, "text": text}
    _write_presets(presets)
    return {"ok": True, "status": 200, "data": {"presets": presets, "saved": name}}


def delete_system_prompt_preset(name: str) -> Dict[str, Any]:
    """Delete a named system-prompt preset. The active system prompt is left
    as-is (deleting a preset never changes what the harness is currently using)."""
    name = (name or "").strip()
    if not name:
        return {"ok": False, "status": 400, "error": "missing_name"}
    presets = _read_presets()
    remaining = [p for p in presets if p["name"].lower() != name.lower()]
    if len(remaining) == len(presets):
        return {"ok": False, "status": 404, "error": "preset_not_found"}
    _write_presets(remaining)
    return {"ok": True, "status": 200, "data": {"presets": remaining, "deleted": name}}


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
            "prompt_preview": _preview(_logged_prompt(ev)),
            "response_preview": _preview(str(ev.get("response_text") or "")),
            "summary": _summary(_logged_prompt(ev)),
            "rag_state": _rag_state(ev),
            "latency_ms": ev.get("latency_ms"),
            "status": _status(cid, queued, verdicts),
        })
    return {"items": items, "count": len(items), "generated_at": iso_now()}


def clear_recent_activity() -> Dict[str, Any]:
    """Clear the observer activity log (every logged LLM call). Submitted verdicts
    and queued items keep their own copies, so the Evaluation queue is unaffected."""
    from aiar.observability import observer
    cleared = observer.clear()
    return {"ok": True, "status": 200,
            "data": {"cleared": cleared, "generated_at": iso_now()}}


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
        "prompt": _logged_prompt(ev),
        "llm_prompt": ev.get("user_prompt"),
        "response": ev.get("response_text"),
        "rag_state": _rag_state(ev),
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
    return {"status": "none", "label": "Not Queued", "score": None, "rating": None}


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
            "prompt": _logged_prompt(ev),
            "llm_prompt": ev.get("user_prompt"),
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


def clear_evaluation_queue(config: Config) -> Dict[str, Any]:
    """Drop every currently queued evaluation item.

    This clears only the queue file; submitted verdicts remain in
    ``verdicts.jsonl`` and still drive the evaluated status in Activity.
    """
    cleared = int(evaluation_queue(config).get("count") or 0)
    try:
        config.queue_file.unlink(missing_ok=True)
    except OSError as exc:
        return {"ok": False, "status": 500, "error": "queue_clear_failed", "data": str(exc)}
    return {"ok": True, "status": 200,
            "data": {"cleared": cleared, "generated_at": iso_now()}}


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

    prompt = _logged_prompt(ev)
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
        "llm_prompt": ev.get("user_prompt"),
        "response": ev.get("response_text"),
    })
    # 2. Reground: feed the correction back into the grounding store.
    rec = grounding_store.record(prompt, verdict, correction=correction)
    return {"ok": True, "status": 200,
            "data": {"call_id": call_id, "rating": rating, "score": score,
                     "regrounded": True, "correction_ts": rec.ts}}
