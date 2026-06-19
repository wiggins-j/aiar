"""LLM transport layer — the only package that talks to Ollama."""

from .ollama_client import (
    MODEL,
    ModelNotPulled,
    OllamaError,
    active_model,
    active_model_ready,
    call_ollama,
    default_model,
    healthcheck,
    installed_model_names,
    list_models,
    resolve_generation_model,
    set_active_model,
)

__all__ = [
    "OllamaError",
    "ModelNotPulled",
    "call_ollama",
    "healthcheck",
    "MODEL",
    "active_model",
    "active_model_ready",
    "default_model",
    "set_active_model",
    "list_models",
    "installed_model_names",
    "resolve_generation_model",
]
