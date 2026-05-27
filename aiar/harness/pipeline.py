"""The prompt harness pipeline: prompt -> retrieve -> (reground) -> answer -> judge.

This is the generic extraction of the upstream ``/eval/prompt`` endpoint:

  1. Retrieve context for the prompt from the RAG (``rag.retriever.get_context``).
  2. Optionally prepend a grounding block of prior corrections (reground).
  3. Answer the prompt with the model, grounded in (1)+(2) when ``rag=True``.
  4. Judge the answer with the LLM-as-judge — ALWAYS against the retrieved
     context, so an ungrounded (rag=false) wrong answer is reliably caught.

Every call is logged by the observer (the watcher GUI tails those logs), so the
returned ``call_id`` lets the GUI mark the exact call for evaluation.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from aiar.llm import call_ollama, OllamaError
from aiar.observability import observer
from aiar import runtime_state

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = (
    "You are a precise assistant. You may be given a Knowledge block retrieved "
    "from a document corpus, then a question. Treat the Knowledge block as the "
    "authoritative source: prefer it over your own memory and do NOT invent "
    "facts. If the provided materials do not contain enough information to "
    "answer confidently, say so plainly instead of answering from general "
    "knowledge. Answer concisely in plain prose."
)

ANSWER_OPTIONS = {"num_predict": 600, "num_ctx": 4096}
ANSWER_TIMEOUT_S = 60

# --------------------------------------------------------------------------
# Active system prompt — HARNESS/ANSWER-PATH SCOPED ONLY.
#
# GUARDRAIL: the operator-editable system prompt governs ONLY the harness/answer
# path (this ``answer_prompt`` non-think branch). It MUST NOT touch any other
# code-defined prompt (judge, query_rewrite, etc.) — those stay code-defined. A
# global override of those would break callers. The think=True path keeps its
# structural REASONING:/ANSWER: contract (ANSWER_THINK_SYSTEM_PROMPT) and is not
# operator-editable in v1.
# Resolution: explicit ``system=`` arg -> process-active -> built-in default.
# --------------------------------------------------------------------------
_active_system_prompt: "Optional[str]" = None  # None -> use the built-in default


def active_system_prompt() -> str:
    """The current harness answer system prompt (active override or built-in)."""
    return str(
        runtime_state.get("active_system_prompt")
        or _active_system_prompt
        or ANSWER_SYSTEM_PROMPT
    )


def set_system_prompt(text: "Optional[str]") -> None:
    """Set the process-global active harness system prompt. Empty/None resets to
    the built-in default."""
    global _active_system_prompt
    text = (text or "").strip()
    _active_system_prompt = text or None
    runtime_state.set_value("active_system_prompt", _active_system_prompt)


def reset_system_prompt() -> None:
    """Restore the built-in default harness system prompt."""
    global _active_system_prompt
    _active_system_prompt = None
    runtime_state.set_value("active_system_prompt", None)

# ``think=True`` demo path — elicit VISIBLE reasoning via a structured
# REASONING:/ANSWER: reply (more reliable across models than native think mode,
# which can consume the whole budget on hidden CoT and return an empty answer).
ANSWER_THINK_SYSTEM_PROMPT = (
    ANSWER_SYSTEM_PROMPT
    + " Before your final answer, SHOW YOUR REASONING: write 2-5 short sentences "
    "of step-by-step reasoning on a line starting exactly with 'REASONING:'. "
    "Then write your final answer on a line starting exactly with 'ANSWER:'. "
    "Include both labels exactly once, in that order."
)
ANSWER_THINK_OPTIONS = {"num_predict": 900, "num_ctx": 4096}
ANSWER_THINK_TIMEOUT_S = 90


def _split_reasoning_answer(text: str) -> "tuple[Optional[str], str]":
    """Split 'REASONING: … ANSWER: …' into (reasoning, answer). If the ANSWER
    marker is missing, reasoning is None and the whole text is the answer."""
    m = re.search(r"ANSWER\s*:", text, flags=re.IGNORECASE)
    if not m:
        return None, text.strip()
    answer = text[m.end():].strip()
    reasoning = re.sub(r"^\s*REASONING\s*:", "", text[: m.start()], flags=re.IGNORECASE).strip()
    return (reasoning or None), answer


def answer_prompt(
    prompt: str,
    *,
    rag: bool = True,
    judge: bool = True,
    think: bool = False,
    reground: Optional[bool] = None,
    top_k: Optional[int] = None,
    context: str = "",
    instance: Optional[str] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    endpoint: str = "/eval/prompt",
) -> Dict[str, Any]:
    """Answer ``prompt`` through the full harness pipeline.

    Returns a dict::

        {
          "answer": str,
          "reasoning": str | None,        # only when think=True
          "verdict": {...} | None,        # the LLM-judge Verdict, when judge=True
          "grounded": bool,               # did the ANSWERER get RAG context?
          "rag_enabled": bool,
          "reground_applied": bool,       # was a grounding block prepended?
          "context_used": bool,
          "latency_ms": int,
          "call_id": str | None,          # observer call_id of the ANSWER call
          "retrieval": {...},             # retrieval features in effect for this
                                          # call (hybrid/rerank/rewrite_mode/top_k/
                                          # fetch_k/grounding_reinjection/rerank_model
                                          # + rag). NOTE: the retrieval pipeline runs
                                          # for the JUDGE regardless of rag; the
                                          # ``rag``/``grounded`` fields say whether
                                          # the ANSWERER was grounded.
        }

    ``rag=False`` blinds the ANSWERER (no retrieved context injected) for a true
    same-pipeline A/B. The JUDGE always sees the retrieved context.

    ``reground`` (None -> honour ``GROUNDING_REINJECTION_ENABLED``): when on, a
    grounding block built from prior corrections is prepended to the answer
    prompt. ``True`` forces it even when the env flag is off.

    ``instance`` selects which named RAG instance to retrieve + ground against
    (None -> the process-active instance; ``"none"`` -> No RAG, skip retrieval).
    ``model`` overrides the active model for this one call (None -> active).
    ``system`` overrides the harness system prompt for this one call (None ->
    the process-active system prompt -> built-in default). Per the guardrail,
    ``system`` governs ONLY this answer call's system prompt.
    """
    from aiar.rag.retriever import get_context
    from aiar.grounding.reinject import grounding_block, reinjection_enabled

    # 1. Retrieve once from the selected instance. The judge always scores
    #    against this; ?rag= only blinds the answerer.
    try:
        retrieved = get_context(prompt, top_k=top_k, instance=instance) or ""
    except Exception:  # pragma: no cover - defensive; RAG must never break here
        retrieved = ""
    answerer_context = retrieved if rag else ""

    # 2. Reground: prepend prior corrections for this prompt's signature, scoped
    #    to the selected instance (corrections never leak across instances).
    force_reground = bool(reground)
    do_reground = reinjection_enabled() if reground is None else bool(reground)
    ground_block = (grounding_block(prompt, force=force_reground, instance=instance)
                    if do_reground else "")

    if rag and not retrieved and not context and not ground_block:
        answer = ("I don't have enough evidence in the available knowledge base "
                  "to answer that confidently.")
        verdict_dict = None
        if judge:
            from aiar.eval.judge import judge_answer
            verdict_dict = judge_answer(prompt, answer, "").to_dict()
        if instance == "none":
            resolved_instance = "none"
        elif instance:
            resolved_instance = instance
        else:
            try:
                from aiar.rag import store as _store
                resolved_instance = _store.active_instance()
            except Exception:
                resolved_instance = "default"
        try:
            from aiar.llm import active_model as _active_model
            resolved_model = model if model is not None else _active_model()
        except Exception:
            resolved_model = model
        if system is not None:
            system_source = "override"
        elif runtime_state.get("active_system_prompt") or _active_system_prompt:
            system_source = "active"
        else:
            system_source = "default"
        try:
            from aiar.rag import settings as rag_settings
            retrieval_cfg = rag_settings.effective()
        except Exception:  # pragma: no cover - defensive
            retrieval_cfg = {}
        if top_k is not None:
            retrieval_cfg["top_k"] = top_k
        retrieval_cfg["rag"] = bool(rag)
        return {
            "answer": answer,
            "reasoning": None,
            "verdict": verdict_dict,
            "grounded": False,
            "rag_enabled": rag,
            "reground_applied": False,
            "context_used": False,
            "latency_ms": 0,
            "call_id": None,
            "instance": resolved_instance,
            "model": resolved_model,
            "system_source": system_source,
            "retrieval": retrieval_cfg,
        }

    # 3. Build the answer prompt.
    parts = []
    if ground_block:
        parts.append(ground_block)
    if answerer_context:
        parts.append(answerer_context)
    if context:
        parts.append(f"ADDITIONAL CONTEXT:\n{context}")
    parts.append(f"QUESTION:\n{prompt}")
    user_prompt = "\n\n".join(parts)

    # System prompt resolution (HARNESS-SCOPED). think=True keeps its structural
    # contract and is NOT operator-editable; the override governs only the
    # non-think answer system prompt.
    if think:
        system_prompt = ANSWER_THINK_SYSTEM_PROMPT
        system_source = "default"
    elif system is not None:
        system_prompt = system
        system_source = "override"
    else:
        resolved = active_system_prompt()
        system_prompt = resolved
        system_source = (
            "active"
            if (runtime_state.get("active_system_prompt") or _active_system_prompt)
            else "default"
        )
    options = ANSWER_THINK_OPTIONS if think else ANSWER_OPTIONS
    timeout = ANSWER_THINK_TIMEOUT_S if think else ANSWER_TIMEOUT_S

    token = observer.set_context(endpoint=endpoint, raw_prompt=prompt)
    capture: Dict[str, Any] = {}
    try:
        raw_answer, latency_ms = call_ollama(
            system_prompt, user_prompt,
            timeout_s=timeout, model=model, options_override=options, format=None,
            capture=capture,
        )
    except OllamaError as exc:
        observer.clear_context(token)
        raise
    finally:
        observer.clear_context(token)

    # The Ollama client captures the emitted call_id directly, so concurrent
    # requests never have to race against the tail of the observer log.
    call_id = capture.get("call_id")

    if think:
        reasoning, answer = _split_reasoning_answer(raw_answer)
    else:
        reasoning, answer = None, raw_answer

    verdict_dict = None
    if judge:
        from aiar.eval.judge import judge_answer
        judge_context = "\n\n".join(p for p in [retrieved, context] if p)
        verdict = judge_answer(prompt, answer, judge_context)
        verdict_dict = verdict.to_dict()

    # Echo the resolved instance/model so the GUI can show which answered.
    if instance == "none":
        resolved_instance = "none"
    elif instance:
        resolved_instance = instance
    else:
        try:
            from aiar.rag import store as _store
            resolved_instance = _store.active_instance()
        except Exception:
            resolved_instance = "default"
    try:
        from aiar.llm import active_model as _active_model
        resolved_model = model if model is not None else _active_model()
    except Exception:
        resolved_model = model

    # Snapshot the retrieval features that were in effect for this answer, so the
    # GUI/metadata shows exactly which frameworks ran (A/B with vs without).
    try:
        from aiar.rag import settings as rag_settings
        retrieval_cfg = rag_settings.effective()
    except Exception:  # pragma: no cover - defensive
        retrieval_cfg = {}
    if top_k is not None:
        retrieval_cfg["top_k"] = top_k
    retrieval_cfg["rag"] = bool(rag)

    return {
        "answer": answer,
        "reasoning": reasoning,
        "verdict": verdict_dict,
        "grounded": bool(answerer_context),
        "rag_enabled": rag,
        "reground_applied": bool(ground_block),
        "context_used": bool(answerer_context),
        "latency_ms": latency_ms,
        "call_id": call_id,
        "instance": resolved_instance,
        "model": resolved_model,
        "system_source": system_source,
        "retrieval": retrieval_cfg,
    }
