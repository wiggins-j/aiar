"""Render past corrections into a prompt block for the next answer.

This is the "reground" half of the loop: given a prompt, look up prior
corrections for its signature and render a compact, capped block that is
prepended to the next answer's prompt, telling the model what it got wrong
before so it self-corrects.

Gated by ``GROUNDING_REINJECTION_ENABLED`` (default OFF) so the prompt shape is
unchanged until you opt in.
"""
from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# Hard caps so a runaway correction history can't blow the prompt budget.
GROUNDING_BLOCK_MAX_CHARS = 1200
GROUNDING_MAX_CORRECTIONS = 3

_ENV_GROUNDING_REINJECTION = "GROUNDING_REINJECTION_ENABLED"


def reinjection_enabled() -> bool:
    """``GROUNDING_REINJECTION_ENABLED`` — default FALSE."""
    raw = os.environ.get(_ENV_GROUNDING_REINJECTION)
    if raw is None or raw == "":
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def grounding_block(prompt: str, *, force: bool = False,
                    instance: "str | None" = None) -> str:
    """Render a compact grounding block from prior corrections for ``prompt``.

    Returns "" when reinjection is off (and ``force`` is False), when nothing is
    recorded, or when no correction carries actionable guidance — so the prompt
    shape is unchanged in all those cases. Pass ``force=True`` to render even
    when the env flag is off (used by the harness when ``reground=True`` is
    explicitly requested for a single call). ``instance`` scopes the correction
    lookup to a RAG instance (per-instance grounding isolation).
    """
    if not reinjection_enabled() and not force:
        return ""
    try:
        from aiar.grounding import store as _grounding
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("grounding block: import failed: %s", exc)
        return ""
    try:
        corrections = _grounding.lookup(prompt, instance=instance)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("grounding block: lookup failed: %s", exc)
        return ""

    # Only surface corrections that carry guidance (skip pure-good provenance).
    actionable = [c for c in corrections if (c.correction or c.reason)]
    if not actionable:
        return ""

    actionable = list(reversed(actionable))[:GROUNDING_MAX_CORRECTIONS]
    lines: List[str] = [
        "GROUNDING — past judgments on similar prompts (correct your answer accordingly):",
    ]
    for c in actionable:
        tags = f" tags={c.failure_tags}" if c.failure_tags else ""
        if c.correction:
            lines.append(f"  - CORRECTION: {c.correction}{tags}")
        elif c.reason:
            lines.append(f"  - PAST ERROR ({c.rating}): {c.reason}{tags}")
    rendered = "\n".join(lines)
    if len(rendered) > GROUNDING_BLOCK_MAX_CHARS:
        rendered = rendered[:GROUNDING_BLOCK_MAX_CHARS]
    return rendered
