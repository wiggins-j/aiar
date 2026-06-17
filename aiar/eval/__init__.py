"""AIAR evaluation: LLM-as-judge + deterministic scoring + A/B runner.

Public surface:
    - ``judge``  : LLM-as-judge -> :class:`Verdict` (rating/reason/tags/confidence)
    - ``scorer`` : deterministic keyword/heuristic scoring of a free-text answer
    - ``runner`` : A/B harness comparing RAG-on vs RAG-off over a case file
    - ``Verdict``: the judge's structured verdict schema
"""

from .schemas import Verdict
from . import judge, scorer, runner

HARNESS_VERSION = "0.2.1"

__all__ = ["Verdict", "judge", "scorer", "runner", "HARNESS_VERSION"]
