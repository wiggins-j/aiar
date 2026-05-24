"""LLM transport layer — the only package that talks to Ollama."""

from .ollama_client import (
    MODEL,
    OllamaError,
    active_model,
    call_ollama,
    default_model,
    healthcheck,
    list_models,
    set_active_model,
)

__all__ = [
    "OllamaError",
    "call_ollama",
    "healthcheck",
    "MODEL",
    "active_model",
    "default_model",
    "set_active_model",
    "list_models",
]
