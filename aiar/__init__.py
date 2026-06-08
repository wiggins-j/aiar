"""AIAR — Artificial Intelligence and RAG.

A model-agnostic (any Qwen via Ollama) RAG + grounding + evaluation +
prompt-harness framework. Pull a model, ingest YOUR documents, run a prompt
harness, rerank + reground, and review/score answers through a web GUI.

This package is deliberately free of any application/domain logic — it is the
generic core extracted to be reused against any corpus.
"""

__version__ = "0.2.0"
__license__ = "Apache-2.0"
