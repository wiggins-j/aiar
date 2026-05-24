"""Deterministic, model-free scoring of a free-text answer against a rubric.

Complements the LLM-as-judge (``aiar.eval.judge``): the judge is a model's
opinion, the scorer is a cheap reproducible signal you fully control. A rubric
is a list of criteria; each awards ``weight`` points:

  * ``match_any``  : award weight if ANY listed substring appears (case-insens).
  * ``forbid_any`` : award weight if NONE of the listed substrings appear.
  * ``min_length`` : award weight if the answer is at least N characters.

Rubrics are plain dicts/JSON — no domain logic baked in. See
``aiar.eval.runner`` for the case-file format that carries a rubric per case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CriterionResult:
    id: str
    weight: int
    earned: int
    matched_on: str = ""


@dataclass
class ScoreResult:
    score: int
    max_score: int
    passed: bool
    breakdown: List[CriterionResult] = field(default_factory=list)

    @property
    def pct(self) -> float:
        return round(self.score / self.max_score, 4) if self.max_score else 0.0

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "max_score": self.max_score,
            "pct": self.pct,
            "passed": self.passed,
            "breakdown": [c.__dict__ for c in self.breakdown],
        }


def score_answer(answer: str, rubric: List[Dict[str, Any]], *, pass_pct: float = 0.6
                 ) -> ScoreResult:
    """Score ``answer`` against ``rubric``. ``passed`` iff score/max >= pass_pct."""
    text = (answer or "").lower()
    breakdown: List[CriterionResult] = []
    total = 0
    earned_total = 0
    for i, crit in enumerate(rubric or []):
        cid = str(crit.get("id") or f"c{i}")
        weight = int(crit.get("weight", 1))
        total += weight
        earned = 0
        matched_on = ""
        if "match_any" in crit:
            for needle in crit["match_any"]:
                if str(needle).lower() in text:
                    earned = weight
                    matched_on = str(needle)
                    break
        elif "forbid_any" in crit:
            hit = next((str(n) for n in crit["forbid_any"] if str(n).lower() in text), "")
            if not hit:
                earned = weight
            else:
                matched_on = f"FORBIDDEN:{hit}"
        elif "min_length" in crit:
            if len(answer or "") >= int(crit["min_length"]):
                earned = weight
                matched_on = f">={crit['min_length']}"
        earned_total += earned
        breakdown.append(CriterionResult(id=cid, weight=weight, earned=earned,
                                         matched_on=matched_on))
    passed = (earned_total / total) >= pass_pct if total else True
    return ScoreResult(score=earned_total, max_score=total, passed=passed,
                       breakdown=breakdown)
