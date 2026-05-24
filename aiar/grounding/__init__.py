"""Grounding memory: the answer -> verdict -> correction -> better-answer loop.

    - ``store``     : persist corrections keyed by a normalized prompt signature
    - ``reinject``  : render past corrections into a prompt block for next time

The store records a *judgment about an answer and how to correct it* (distinct
from a vector RAG corpus, which records source documents). On the next answer
for a matching prompt, ``reinject.grounding_block`` looks up the corrections and
renders a compact block telling the model what it got wrong before.
"""

from .store import Correction, GroundingStore, record, lookup, normalize_signature
from .reinject import grounding_block

__all__ = [
    "Correction",
    "GroundingStore",
    "record",
    "lookup",
    "normalize_signature",
    "grounding_block",
]
