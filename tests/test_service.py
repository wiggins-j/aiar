from __future__ import annotations

import pytest

from aiar.harness import pipeline, service
from aiar.llm import ollama_client
from aiar.rag import store
from web import aggregator


@pytest.fixture
def shared_runtime_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AIAR_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("AIAR_DB_PATH", str(tmp_path / "knowledge"))
    monkeypatch.delenv("RAG_INSTANCE", raising=False)
    monkeypatch.delenv("AIAR_RUNTIME_STATE_FILE", raising=False)
    store.reset_for_testing(base=tmp_path)
    ollama_client.reset_for_testing()
    pipeline.reset_system_prompt()
    yield tmp_path
    pipeline.reset_system_prompt()
    ollama_client.reset_for_testing()
    store.reset_for_testing()


def test_service_prompt_passes_endpoint_and_request_options(monkeypatch, shared_runtime_state):
    captured = {}

    def fake_answer_prompt(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {"answer": "ok", "grounded": True}

    monkeypatch.setattr(service, "answer_prompt", fake_answer_prompt)
    monkeypatch.setattr(service, "resolve_generation_model", lambda m=None: m)
    req = service.ServicePromptRequest(
        service_name="LocalApp",
        prompt="hello",
        instance="alpha",
        model="qwen3.5:9b",
        system="be exact",
        rag=False,
        judge=True,
        think=True,
        reground=True,
        top_k=9,
        metadata={"job": "test"},
    )

    out = service.service_prompt(req)
    assert out["service_name"] == "LocalApp"
    assert out["service_metadata"] == {"job": "test"}
    assert captured["prompt"] == "hello"
    assert captured["kwargs"]["instance"] == "alpha"
    assert captured["kwargs"]["model"] == "qwen3.5:9b"
    assert captured["kwargs"]["system"] == "be exact"
    assert captured["kwargs"]["rag"] is False
    assert captured["kwargs"]["judge"] is True
    assert captured["kwargs"]["think"] is True
    assert captured["kwargs"]["reground"] is True
    assert captured["kwargs"]["top_k"] == 9
    assert captured["kwargs"]["endpoint"] == "/services/prompt"


def test_services_meta_requests_unfiltered_model_list(monkeypatch, shared_runtime_state):
    called = {}

    def fake_list_models(*, show_all=False):
        called["show_all"] = show_all
        return [{"name": "qwen3.5:9b"}]

    monkeypatch.setattr(service, "healthcheck", lambda: True)
    monkeypatch.setattr(service, "list_models", fake_list_models)
    monkeypatch.setattr(service, "active_model", lambda: "qwen3.5:9b")
    monkeypatch.setattr(service.store, "health", lambda: {"store_ready": True, "active_instance": "default"})
    monkeypatch.setattr(service.store, "list_instances", lambda: [{"name": "default", "active": True, "chunk_count": 0}])

    out = service.services_meta()
    assert out["ok"] is True
    assert called["show_all"] is True
    assert out["available_models"] == [{"name": "qwen3.5:9b"}]


def test_runtime_state_shared_between_settings_and_service(shared_runtime_state, monkeypatch):
    monkeypatch.setattr(
        ollama_client,
        "list_models",
        lambda show_all=False: [
            {"name": "qwen3.5:9b"},
            {"name": "qwen2.5-coder:1.5b"},
        ],
    )
    monkeypatch.setattr(ollama_client, "healthcheck", lambda: True)
    store.create_instance("alpha")

    assert aggregator.set_active_model("qwen3.5:9b")["ok"] is True
    assert aggregator.set_active_rag("alpha")["ok"] is True
    aggregator.set_system_prompt("You are terse.")

    out = service.services_meta()
    assert out["active_model"] == "qwen3.5:9b"
    assert out["rag"]["active_instance"] == "alpha"
    assert out["system_prompt"]["has_override"] is True
    assert out["system_prompt"]["label"] == "Custom override"
