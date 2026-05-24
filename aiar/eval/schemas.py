"""Evaluation schemas.

Kept dependency-free (plain dataclasses, no pydantic) so the AIAR core runs on
a stock Python 3.10+ without extra installs. The single shape that matters is
:class:`Verdict` — an LLM-as-judge quality judgment of one (prompt, response)
pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

_VALID_RATINGS = ("good", "bad", "partial")
_VALID_CONFIDENCE = ("low", "medium", "high")


@dataclass
class Verdict:
    """An LLM-as-judge quality verdict on a single (prompt, response) pair.

    - ``rating``: ``good`` | ``partial`` | ``bad``. ``partial`` covers answers
      that are directionally right but incomplete or hedged.
    - ``reason``: free-text justification — should name the specific error.
    - ``failure_tags``: short machine-matchable tags (e.g.
      ``hallucinated_fact``, ``incomplete``) so grounding lookups can cluster
      similar failures.
    - ``confidence``: the judge's own confidence in this verdict.
    """

    rating: str = "bad"
    reason: str = ""
    failure_tags: List[str] = field(default_factory=list)
    confidence: str = "medium"

    def __post_init__(self) -> None:
        if self.rating not in _VALID_RATINGS:
            self.rating = "bad"
        if self.confidence not in _VALID_CONFIDENCE:
            self.confidence = "medium"
        if not isinstance(self.failure_tags, list):
            self.failure_tags = []

    @classmethod
    def from_dict(cls, d: dict) -> "Verdict":
        d = d or {}
        tags = d.get("failure_tags") or []
        if not isinstance(tags, list):
            tags = []
        return cls(
            rating=str(d.get("rating") or "bad"),
            reason=str(d.get("reason") or ""),
            failure_tags=[str(t) for t in tags],
            confidence=str(d.get("confidence") or "medium"),
        )

    def to_dict(self) -> dict:
        return {
            "rating": self.rating,
            "reason": self.reason,
            "failure_tags": list(self.failure_tags),
            "confidence": self.confidence,
        }

    # Convenience alias so callers can use the pydantic-style name.
    def model_dump(self) -> dict:
        return self.to_dict()
