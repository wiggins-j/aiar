"""LLM-as-judge for arbitrary (prompt, response[, context]) answer quality.

Given a ``(prompt, response[, context])`` triple this asks the model to rate
the response ``good`` / ``bad`` / ``partial`` **with a reason why** plus
machine-matchable ``failure_tags`` and a ``confidence``. Returns a
:class:`aiar.eval.schemas.Verdict`.

The judge uses a colder config than the answerer (temperature 0, larger
context) so the verdict is stable and it can read a long answer. When a
``context`` block (the retrieved RAG context) is supplied the judge treats it
as the authoritative ground truth to score against — that is what makes an
ungrounded wrong answer reliably catchable.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Callable, Optional, Tuple

from aiar.llm import call_ollama, OllamaError

from .schemas import Verdict

logger = logging.getLogger(__name__)

_JUDGE_OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "num_predict": 400,
    "num_ctx": 4096,
}

_JUDGE_TIMEOUT_S = int(os.environ.get("EVAL_JUDGE_TIMEOUT_S", "30"))

_JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of an AI assistant's answers.

You are given the PROMPT the assistant was asked and the assistant's RESPONSE.
You may also be given a CONTEXT block of retrieved source material — when
present, treat it as the authoritative ground truth and judge the RESPONSE
against it. Judge whether the response is correct, complete, and safe.

If the response makes a factual claim that contradicts the CONTEXT, or that is
plainly wrong, say exactly which claim is wrong.

Reply with STRICT JSON only, matching this schema:
{
  "rating": "good" | "bad" | "partial",
  "reason": "<one or two sentences naming the specific error or why it is good>",
  "failure_tags": ["<short_snake_case_tag>", ...],
  "confidence": "low" | "medium" | "high"
}

Rating guide:
- "good": correct, complete, and safe.
- "partial": directionally right but incomplete, hedged, or missing a key point.
- "bad": contains a factual error, is unsafe, or fails to answer.

Use "failure_tags" for machine-matchable error classes, e.g.
"hallucinated_fact", "contradicts_context", "incomplete", "off_topic". Use []
when rating is "good". Output the JSON object only — no prose, no markdown."""


def _build_judge_user_prompt(prompt: str, response: str, context: str = "") -> str:
    parts = [f"PROMPT:\n{prompt}", f"RESPONSE:\n{response}"]
    if context:
        parts.append(f"CONTEXT:\n{context}")
    parts.append("Judge the RESPONSE now. Output the JSON verdict object only.")
    return "\n\n".join(parts)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_verdict_json(raw: str) -> Optional[dict]:
    """Parse the verdict object from a raw model string (json.loads, then a
    fallback outermost ``{...}`` grab). Returns None if neither yields a dict."""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except (ValueError, TypeError):
        pass
    m = _JSON_OBJ_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            return None
    return None


# Injectable LLM seam — matches ``call_ollama``'s (text, latency_ms) return so
# tests can mock the model without a live Ollama.
LlmCaller = Callable[..., Tuple[str, int]]


def judge_answer(
    prompt: str,
    response: str,
    context: str = "",
    *,
    llm_caller: Optional[LlmCaller] = None,
) -> Verdict:
    """Judge a single answer, returning a structured :class:`Verdict`.

    On any failure (LLM error, unparseable output) a conservative ``bad``
    verdict is returned rather than raising — the caller always gets a usable
    verdict.
    """
    caller = llm_caller or call_ollama
    user_prompt = _build_judge_user_prompt(prompt, response, context)

    try:
        raw, _latency = caller(
            _JUDGE_SYSTEM_PROMPT,
            user_prompt,
            timeout_s=_JUDGE_TIMEOUT_S,
            options_override=_JUDGE_OPTIONS,
        )
    except OllamaError as exc:
        logger.warning("judge_answer: LLM call failed: %s", exc)
        return Verdict(rating="bad", reason=f"judge could not run: {exc}",
                       failure_tags=["judge_error"], confidence="low")

    obj = _extract_verdict_json(raw)
    if obj is None:
        logger.info("judge_answer: unparseable verdict: %r", (raw or "")[:200])
        return Verdict(rating="bad", reason="judge returned no parseable JSON verdict",
                       failure_tags=["judge_unparseable"], confidence="low")

    return Verdict.from_dict(obj)
