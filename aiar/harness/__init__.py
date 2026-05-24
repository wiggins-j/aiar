"""AIAR prompt harness: prompt -> retrieve -> (reground) -> answer -> judge.

    - ``answer_prompt``  : the core in-process pipeline (returns answer+verdict)
    - ``service``        : an optional FastAPI service exposing POST /eval/prompt
    - CLI: ``python -m aiar.harness "your prompt here"``
"""

from .pipeline import (
    active_system_prompt,
    answer_prompt,
    reset_system_prompt,
    set_system_prompt,
)

__all__ = [
    "answer_prompt",
    "active_system_prompt",
    "set_system_prompt",
    "reset_system_prompt",
]
