"""F098 — active-model reliability: readiness markers, structured model_not_pulled
error, runtime repoint, and optional auto-fallback. Ollama is faked via
list_models, and runtime-state is isolated to a tmp dir, so these run offline."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aiar import runtime_state
from aiar.llm import ollama_client
from aiar.harness import service

AUTH = {"Authorization": "Bearer secret"}

PULLED = [
    {"name": "gemma3:27b", "size_bytes": 17_000_000_000, "family": "gemma3"},
    {"name": "qwen3.5:9b", "size_bytes": 6_000_000_000, "family": "qwen3"},
    {"name": "nomic-embed-text", "size_bytes": 270_000_000, "family": "nomic-bert"},
]


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path):
    # Isolate runtime-state to tmp so active_model() reads/writes a real (but
    # throwaway) store — no monkeypatching of runtime_state.get, so live repoint
    # via set_active_model actually takes effect.
    ollama_client.reset_for_testing()
    runtime_state.reset_for_testing(base=tmp_path)
    yield
    ollama_client.reset_for_testing()


def _set_active(name):
    runtime_state.set_value("active_model", name)


def _fake_models(monkeypatch, models=PULLED):
    monkeypatch.setattr(ollama_client, "list_models",
                        lambda *, show_all=False: models)


# --- resolver / readiness (unit) ------------------------------------------

def test_active_model_ready_false_when_not_pulled(monkeypatch):
    _fake_models(monkeypatch)
    _set_active("qwen2.5:7b")  # not in PULLED
    assert ollama_client.active_model_ready() is False


def test_active_model_ready_true_when_pulled(monkeypatch):
    _fake_models(monkeypatch)
    _set_active("qwen3.5:9b")
    assert ollama_client.active_model_ready() is True


def test_resolve_raises_model_not_pulled(monkeypatch):
    _fake_models(monkeypatch)
    _set_active("qwen2.5:7b")
    with pytest.raises(ollama_client.ModelNotPulled) as ei:
        ollama_client.resolve_generation_model(None)
    assert ei.value.model == "qwen2.5:7b"
    assert "qwen3.5:9b" in ei.value.available


def test_resolve_passthrough_when_ollama_unreachable(monkeypatch):
    # Empty installed set = unreachable -> don't mislabel as not-pulled.
    _fake_models(monkeypatch, models=[])
    _set_active("qwen2.5:7b")
    assert ollama_client.resolve_generation_model(None) == "qwen2.5:7b"


def test_resolve_explicit_override_must_be_pulled(monkeypatch):
    _fake_models(monkeypatch)
    _set_active("qwen3.5:9b")
    with pytest.raises(ollama_client.ModelNotPulled):
        ollama_client.resolve_generation_model("does-not-exist:1b")


def test_fallback_picks_smallest_non_embedding(monkeypatch):
    monkeypatch.setenv("AIAR_ACTIVE_MODEL_FALLBACK", "auto")
    _fake_models(monkeypatch)
    _set_active("qwen2.5:7b")
    # smallest non-embedding is qwen3.5:9b (6GB) — NOT nomic-embed-text (270MB).
    assert ollama_client.resolve_generation_model(None) == "qwen3.5:9b"
    assert ollama_client.active_model() == "qwen3.5:9b"  # persisted


def test_fallback_not_applied_to_explicit_override(monkeypatch):
    monkeypatch.setenv("AIAR_ACTIVE_MODEL_FALLBACK", "auto")
    _fake_models(monkeypatch)
    _set_active("qwen3.5:9b")
    with pytest.raises(ollama_client.ModelNotPulled):
        ollama_client.resolve_generation_model("explicit-missing:1b")


# --- HTTP surface ----------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(service, "healthcheck", lambda: True)
    monkeypatch.setattr(service.store, "health", lambda: {"store_ready": True})
    monkeypatch.setattr(service.store, "init", lambda: None)
    monkeypatch.setattr(service, "list_models", lambda *, show_all=False: PULLED)
    monkeypatch.setattr(ollama_client, "list_models", lambda *, show_all=False: PULLED)
    return TestClient(service.app)


def test_healthz_reports_active_model_ready_false(client):
    _set_active("qwen2.5:7b")
    h = client.get("/healthz").json()
    assert h["active_model"] == "qwen2.5:7b"
    assert h["active_model_ready"] is False


def test_capabilities_generation_false_when_unpulled(client):
    _set_active("qwen2.5:7b")
    m = client.get("/capabilities").json()
    assert m["features"]["generation"] is False


def test_services_prompt_returns_structured_model_not_pulled(client):
    _set_active("qwen2.5:7b")  # not pulled
    r = client.post("/services/prompt", json={"service_name": "X", "prompt": "hi"})
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["code"] == "model_not_pulled"
    assert d["model"] == "qwen2.5:7b"
    assert "qwen3.5:9b" in d["available_models"]


def test_set_model_repoints_live(client, monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    _set_active("qwen2.5:7b")
    r = client.post("/services/model", json={"model": "qwen3.5:9b"}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["active_model"] == "qwen3.5:9b"
    assert r.json()["active_model_ready"] is True
    assert ollama_client.active_model() == "qwen3.5:9b"  # really repointed


def test_set_model_unpulled_returns_409(client, monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    r = client.post("/services/model", json={"model": "ghost:1b"}, headers=AUTH)
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["code"] == "model_not_pulled"
    assert d["model"] == "ghost:1b"


def test_set_model_requires_token(client, monkeypatch):
    monkeypatch.setenv("AIAR_SERVICE_TOKEN", "secret")
    r = client.post("/services/model", json={"model": "qwen3.5:9b"})  # no header
    assert r.status_code == 401
