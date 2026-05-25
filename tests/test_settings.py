"""Tests for the Settings + switching slice (multi-RAG instances, model switch,
system-prompt editor) — the AIAR born-instance-aware backend (Group F) and the
AIAR Settings page + in-process routes (Group E).

These use real chromadb (installed) for instance isolation, and pin a tmp DB
path + registry base per test so the process-global store never pollutes across
tests. Every test that touches the store calls ``store.reset_for_testing()``.

Run:  python -m pytest tests/test_settings.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiar.rag import store
from aiar.rag.ingest import Chunk


# ---------------------------------------------------------------------------
# Fixtures — pin the store to a tmp DB + registry so the process-global state
# never touches ~/.aiar and never leaks between tests.
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """A store re-initialised against a tmp ChromaDB + registry base."""
    monkeypatch.setenv("AIAR_DB_PATH", str(tmp_path / "knowledge"))
    monkeypatch.delenv("AIAR_CORPUS", raising=False)
    monkeypatch.delenv("RAG_INSTANCE", raising=False)
    store.reset_for_testing(base=tmp_path)
    yield store
    store.reset_for_testing()


def _chunk(source: str, text: str, idx: int = 0, title: str = "Doc",
           category: str = "general") -> Chunk:
    return Chunk(source=source, title=title, chunk_index=idx, text=text,
                 category=category)


# ===========================================================================
# F1 — store instance map, list_instances/set_active/active_instance, isolation
# ===========================================================================

def test_aiar_default_instance(fresh_store):
    """A born-instance-aware store resolves to ``default`` with no arg, no env,
    no toggle — and the default collection honours AIAR_CORPUS (preserving any
    existing corpus)."""
    assert fresh_store.active_instance() == "default"
    # Writing with no explicit instance is rejected (write isolation)...
    with pytest.raises(TypeError):
        fresh_store.add([_chunk("a.md", "hello")])  # missing instance= kwarg
    # ...but the read path resolves to default.
    assert fresh_store.chunk_count() == 0
    assert fresh_store.chunk_count(instance="default") == 0


def test_aiar_ingest_cli_writes_to_active_and_named_instance(fresh_store, tmp_path):
    """Regression: ``python -m aiar.rag.ingest <folder>`` must not crash. F1 made
    ``store.add(instance=)`` required; the CLI must resolve + pass an instance.
    Default ingest -> the active ('default') instance; ``--instance`` -> a named
    one (auto-created on first ingest); ``--dry-run`` writes nothing."""
    from aiar.rag import ingest as ingest_mod

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text(
        "AIAR ingest smoke: the quick brown fox jumps over the lazy dog. " * 5,
        encoding="utf-8")

    # Default ingest resolves to the active instance ("default") and writes.
    assert ingest_mod.main([str(docs)]) == 0
    assert fresh_store.chunk_count(instance="default") > 0

    # --instance routes into a named instance (created on first ingest), isolated.
    assert ingest_mod.main([str(docs), "--instance", "tesla"]) == 0
    assert fresh_store.chunk_count(instance="tesla") > 0

    # --dry-run never writes.
    before = fresh_store.chunk_count(instance="tesla")
    docs2 = tmp_path / "docs2"
    docs2.mkdir()
    (docs2 / "x.md").write_text("another doc for the dry-run preview path", encoding="utf-8")
    assert ingest_mod.main([str(docs2), "--instance", "tesla", "--dry-run"]) == 0
    assert fresh_store.chunk_count(instance="tesla") == before


def test_aiar_ingest_publishes_instance(fresh_store, tmp_path):
    """A successful ingest marks the instance 'published' (not left as draft) so
    the GUI doesn't show a permanent [draft] badge."""
    from aiar.rag import ingest as ingest_mod
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "n.md").write_text("AIAR publish-on-ingest test. " * 30, encoding="utf-8")
    assert ingest_mod.main([str(docs), "--instance", "pubtest"]) == 0
    rows = {r["name"]: r for r in fresh_store.list_instances()}
    assert rows["pubtest"]["status"] == "published"


def test_aiar_add_batches_over_cap(fresh_store, monkeypatch):
    """store.add() writes in batches so an ingest larger than ChromaDB's per-call
    cap succeeds (regression: a single add() of >5461 chunks used to fail)."""
    monkeypatch.setattr(fresh_store, "_MAX_ADD_BATCH", 2)
    fresh_store.create_instance("big")
    chunks = [_chunk(f"d{i}.md", f"unique chunk text number {i} alpha beta", idx=i)
              for i in range(5)]
    assert fresh_store.add(chunks, instance="big") == 5   # 5 items, batch size 2
    assert fresh_store.chunk_count(instance="big") == 5


def test_aiar_instance_isolation(fresh_store):
    """Ingest into two named instances; queries + counts never cross-talk."""
    fresh_store.create_instance("alpha")
    fresh_store.create_instance("beta")
    n = fresh_store.add([_chunk("a.md", "alpha apples are crunchy", title="Alpha")],
                        instance="alpha")
    assert n == 1
    fresh_store.add([_chunk("b.md", "beta bananas are soft", title="Beta")],
                    instance="beta")

    assert fresh_store.chunk_count(instance="alpha") == 1
    assert fresh_store.chunk_count(instance="beta") == 1
    assert fresh_store.chunk_count(instance="default") == 0

    hits = fresh_store.query_scored("apples", n_results=3, instance="alpha")
    assert hits and all("apple" in h.text.lower() for h in hits)
    # beta must NOT surface alpha's apples
    beta_hits = fresh_store.query_scored("apples", n_results=3, instance="beta")
    assert all("alpha apples" not in h.text.lower() for h in beta_hits)


def test_aiar_list_instances(fresh_store):
    """list_instances returns descriptors with name/status/chunk_count/active."""
    fresh_store.create_instance("alpha", display_name="Alpha Corpus")
    fresh_store.add([_chunk("a.md", "alpha content")], instance="alpha")
    rows = fresh_store.list_instances()
    by_name = {r["name"]: r for r in rows}
    assert "default" in by_name and "alpha" in by_name
    assert by_name["default"]["active"] is True
    assert by_name["default"]["display_name"] == "Example RAG"
    assert by_name["alpha"]["active"] is False
    assert by_name["alpha"]["display_name"] == "Alpha Corpus"
    assert by_name["alpha"]["chunk_count"] == 1
    assert by_name["alpha"]["status"] in ("draft", "published")


def test_aiar_set_active_toggle(fresh_store):
    """set_active flips the process-global active instance with no restart."""
    fresh_store.create_instance("alpha")
    fresh_store.add([_chunk("a.md", "alpha content")], instance="alpha")
    assert fresh_store.active_instance() == "default"
    fresh_store.set_active("alpha")
    assert fresh_store.active_instance() == "alpha"
    # An instance-less count now resolves to alpha.
    assert fresh_store.chunk_count() == 1
    fresh_store.set_active("default")
    assert fresh_store.active_instance() == "default"


def test_aiar_set_active_rejects_unknown(fresh_store):
    with pytest.raises(ValueError):
        fresh_store.set_active("does-not-exist")


def test_aiar_delete_instance(fresh_store):
    """delete_instance drops the collection + registry entry; the instance no
    longer lists, and a fresh handle to its name starts empty (no stale data)."""
    fresh_store.create_instance("alpha")
    fresh_store.add([_chunk("a.md", "alpha content to be deleted")], instance="alpha")
    assert fresh_store.chunk_count(instance="alpha") == 1

    res = fresh_store.delete_instance("alpha")
    assert res == {"deleted": "alpha", "active": "default"}
    names = {r["name"] for r in fresh_store.list_instances()}
    assert "alpha" not in names
    # Re-resolving the name auto-creates a fresh, empty collection (not the old data).
    assert fresh_store.chunk_count(instance="alpha") == 0


def test_aiar_delete_active_instance_resets_to_default(fresh_store):
    """Deleting the active instance falls back to ``default`` so reads stay valid."""
    fresh_store.create_instance("alpha")
    fresh_store.set_active("alpha")
    assert fresh_store.active_instance() == "alpha"
    res = fresh_store.delete_instance("alpha")
    assert res["active"] == "default"
    assert fresh_store.active_instance() == "default"


def test_aiar_delete_instance_guards(fresh_store):
    """The default instance and the ``none`` sentinel are not deletable; an
    unknown name raises rather than silently succeeding."""
    with pytest.raises(ValueError):
        fresh_store.delete_instance("default")
    with pytest.raises(ValueError):
        fresh_store.delete_instance("none")
    with pytest.raises(ValueError):
        fresh_store.delete_instance("does-not-exist")


def test_aiar_delete_instance_cleans_grounding(fresh_store, tmp_path, monkeypatch):
    """A full delete also purges the instance's own grounding corrections subdir,
    while the shared grounding root and global (flat) corrections survive."""
    from aiar.grounding import store as grounding_store
    from aiar.eval.schemas import Verdict
    monkeypatch.setenv("GROUNDING_BASE_DIR", str(tmp_path))
    v = Verdict(rating="bad", reason="wrong")
    grounding_store.record("how long?", v, correction="alpha ans",
                           base=tmp_path, instance="alpha")
    grounding_store.record("global q", v, correction="global ans", base=tmp_path)
    inst_dir = tmp_path / "grounding" / "alpha"
    assert inst_dir.is_dir()

    fresh_store.create_instance("alpha")
    fresh_store.delete_instance("alpha")

    assert not inst_dir.exists()                              # per-RAG state purged
    assert (tmp_path / "grounding").is_dir()                  # shared root kept
    assert grounding_store.lookup("global q", base=tmp_path) != []  # globals intact


def test_aiar_registry_persists_and_self_heals(fresh_store, tmp_path):
    """A descriptor is written to registry.json; a rag_* collection with no
    registry entry is back-filled as a draft on the next init (self-heal)."""
    fresh_store.create_instance("alpha", display_name="Alpha")
    reg_path = tmp_path / "knowledge" / "registry.json"
    assert reg_path.exists()
    data = json.loads(reg_path.read_text())
    assert "alpha" in data

    # Create a stray rag_* collection directly, then re-init: it must surface.
    fresh_store._client.get_or_create_collection(  # type: ignore[attr-defined]
        name="rag_orphan", metadata={"hnsw:space": "cosine"})
    fresh_store.reset_for_testing(base=tmp_path)
    names = {r["name"] for r in fresh_store.list_instances()}
    assert "orphan" in names


def test_aiar_default_instance_label_migrates_from_legacy_default(tmp_path):
    from aiar.rag import instances

    reg_path = tmp_path / "knowledge" / "registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps({
        "default": {
            "name": "default",
            "display_name": "Default",
            "collection": "aiar",
            "status": "published",
        }
    }), encoding="utf-8")

    registry = instances.Registry(tmp_path, default_collection="aiar")
    assert registry.get("default").display_name == "Example RAG"


# ===========================================================================
# F2 — retriever threads instance; ``none`` skips retrieval
# ===========================================================================

def test_aiar_per_request_instance(fresh_store):
    """get_context(query, instance='alpha') retrieves from alpha regardless of
    the active instance."""
    from aiar.rag import retriever
    fresh_store.create_instance("alpha")
    fresh_store.create_instance("beta")
    fresh_store.add([_chunk("a.md", "the alpha widget ships in two days")],
                    instance="alpha")
    fresh_store.add([_chunk("b.md", "the beta gadget ships in five days")],
                    instance="beta")
    # active is default (empty) — explicit instance still retrieves.
    ctx = retriever.get_context("widget shipping", instance="alpha", rewrite=False)
    assert "alpha widget" in ctx
    ctx_beta = retriever.get_context("gadget shipping", instance="beta", rewrite=False)
    assert "beta gadget" in ctx_beta
    assert "alpha widget" not in ctx_beta


def test_aiar_no_rag_skips(fresh_store):
    """instance='none' skips retrieval entirely and returns '' (mirrors
    rag=false answerer-blinding)."""
    from aiar.rag import retriever
    fresh_store.create_instance("alpha")
    fresh_store.add([_chunk("a.md", "alpha content here")], instance="alpha")
    fresh_store.set_active("alpha")
    # Active is alpha with content, but 'none' must return "".
    assert retriever.get_context("content", instance="none", rewrite=False) == ""


def test_aiar_query_rewrite_instance_kwarg(fresh_store, monkeypatch):
    """transform accepts instance= (domain-generic — descriptor prompts optional)
    and uses a per-instance rewrite system prompt when the descriptor sets one."""
    from aiar.rag import query_rewrite
    captured = {}

    def fake_call(system, user, **kwargs):
        captured["system"] = system
        return ("rewritten query", 5)

    monkeypatch.setenv("RAG_QUERY_REWRITE_MODE", "rewrite")
    monkeypatch.setattr("aiar.llm.call_ollama", fake_call)
    fresh_store.create_instance(
        "alpha",
        query_rewrite={"rewrite_system": "ALPHA-SPECIFIC REWRITE PROMPT",
                       "hyde_system": "ALPHA HYDE"},
    )
    out = query_rewrite.transform("what is x", instance="alpha")
    assert out == "rewritten query"
    assert captured["system"] == "ALPHA-SPECIFIC REWRITE PROMPT"
    # default instance falls back to the generic built-in.
    captured.clear()
    query_rewrite.transform("what is y", instance="default")
    assert "ALPHA-SPECIFIC" not in captured["system"]


# ===========================================================================
# F1 coupling-5b — fusion get_by_ids scoped to instance
# ===========================================================================

# ===========================================================================
# F3 — active-model layer in aiar/llm/ollama_client.py
# ===========================================================================

@pytest.fixture
def fresh_llm(monkeypatch):
    from aiar.llm import ollama_client
    monkeypatch.setenv("OLLAMA_MODEL", "qwen-test:1b")
    ollama_client.reset_for_testing()
    yield ollama_client
    ollama_client.reset_for_testing()


def test_aiar_active_model_defaults_to_env_seed(fresh_llm):
    assert fresh_llm.active_model() == "qwen-test:1b"
    assert fresh_llm.default_model() == "qwen-test:1b"
    assert fresh_llm.MODEL == "qwen-test:1b"  # back-compat alias kept


def test_aiar_set_active_model_validates(fresh_llm, monkeypatch):
    monkeypatch.setattr(
        fresh_llm, "list_models",
        lambda **_: [{"name": "qwen-test:1b"}, {"name": "qwen-big:9b"}])
    fresh_llm.set_active_model("qwen-big:9b")
    assert fresh_llm.active_model() == "qwen-big:9b"
    with pytest.raises(ValueError):
        fresh_llm.set_active_model("not-installed:42b")
    assert fresh_llm.active_model() == "qwen-big:9b"  # unchanged on reject


def test_aiar_list_models_parses_and_filters(monkeypatch):
    from aiar.llm import ollama_client

    class FakeResp:
        status_code = 200

        def json(self):
            return {"models": [
                {"name": "qwen3.5:9b", "size": 100, "details": {"family": "qwen"}},
                {"name": "llama3:8b", "size": 200, "details": {"family": "llama"}},
            ]}

    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp())
    monkeypatch.setenv("MODEL_LIST_PREFIXES", "qwen")
    models = ollama_client.list_models()
    names = {m["name"] for m in models}
    assert "qwen3.5:9b" in names and "llama3:8b" not in names
    assert all("size_bytes" in m and "family" in m for m in models)
    # show-all escape
    monkeypatch.setenv("MODEL_LIST_PREFIXES", "")
    names_all = {m["name"] for m in ollama_client.list_models()}
    assert "llama3:8b" in names_all


def test_aiar_list_models_unreachable_returns_empty(monkeypatch):
    import requests
    from aiar.llm import ollama_client

    def boom(*a, **k):
        raise requests.RequestException("down")

    monkeypatch.setattr("requests.get", boom)
    assert ollama_client.list_models() == []


def test_aiar_call_ollama_resolves_active_model(fresh_llm, monkeypatch):
    sent = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "ok", "prompt_eval_count": 1, "eval_count": 1}

    def fake_post(url, json=None, timeout=None):
        sent["model"] = json["model"]
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(
        fresh_llm, "list_models",
        lambda **_: [{"name": "qwen-test:1b"}, {"name": "qwen-big:9b"}])
    # model=None -> resolves active
    fresh_llm.call_ollama("sys", "user", model=None)
    assert sent["model"] == "qwen-test:1b"
    fresh_llm.set_active_model("qwen-big:9b")
    fresh_llm.call_ollama("sys", "user")  # default arg is now the sentinel
    assert sent["model"] == "qwen-big:9b"
    # explicit model arg overrides active
    fresh_llm.call_ollama("sys", "user", model="qwen-test:1b")
    assert sent["model"] == "qwen-test:1b"


# ===========================================================================
# F3 — per-instance grounding
# ===========================================================================

def test_aiar_per_instance_grounding(tmp_path):
    from aiar.grounding import store as grounding_store
    from aiar.eval.schemas import Verdict
    v = Verdict(rating="bad", reason="wrong", failure_tags=["x"])
    grounding_store.record("how long?", v, correction="alpha answer",
                           base=tmp_path, instance="alpha")
    # alpha sees it; beta does not (corrections never leak across instances).
    a = grounding_store.lookup("how long?", base=tmp_path, instance="alpha")
    b = grounding_store.lookup("how long?", base=tmp_path, instance="beta")
    assert a and a[-1].correction == "alpha answer"
    assert b == []


# ===========================================================================
# F3 — harness resolves instance/model/system
# ===========================================================================

def _fake_harness_llm(monkeypatch, sent):
    """Patch the harness's call_ollama + judge to be deterministic (no network)."""
    def fake_call(system_prompt, user_prompt, **kwargs):
        sent["system"] = system_prompt
        sent["user"] = user_prompt
        sent["model"] = kwargs.get("model")
        cap = kwargs.get("capture")
        if cap is not None:
            cap["thinking"] = None
        return ("the answer", 12)

    monkeypatch.setattr("aiar.harness.pipeline.call_ollama", fake_call)


def test_aiar_harness_system_override(monkeypatch):
    from aiar.harness import pipeline
    pipeline.reset_system_prompt()
    sent = {}
    _fake_harness_llm(monkeypatch, sent)
    pipeline.answer_prompt("q", rag=False, judge=False,
                           system="CUSTOM SYSTEM PROMPT")
    assert sent["system"] == "CUSTOM SYSTEM PROMPT"
    # active system prompt (process-global) is used when no override is passed
    pipeline.set_system_prompt("ACTIVE SYSTEM PROMPT")
    pipeline.answer_prompt("q", rag=False, judge=False)
    assert sent["system"] == "ACTIVE SYSTEM PROMPT"
    # reset restores the built-in default
    pipeline.reset_system_prompt()
    pipeline.answer_prompt("q", rag=False, judge=False)
    assert sent["system"] == pipeline.ANSWER_SYSTEM_PROMPT


def test_aiar_harness_model_override(monkeypatch):
    from aiar.harness import pipeline
    sent = {}
    _fake_harness_llm(monkeypatch, sent)
    pipeline.answer_prompt("q", rag=False, judge=False, model="qwen-x:9b")
    assert sent["model"] == "qwen-x:9b"


def test_aiar_harness_instance_select(fresh_store, monkeypatch):
    from aiar.harness import pipeline
    fresh_store.create_instance("alpha")
    fresh_store.add([_chunk("a.md", "alpha-only sentinel fact about ferrets")],
                    instance="alpha")
    sent = {}
    _fake_harness_llm(monkeypatch, sent)
    pipeline.answer_prompt("ferrets", rag=True, judge=False, instance="alpha")
    assert "alpha-only sentinel" in sent["user"]
    # instance="none" blinds the answerer even though alpha has data
    sent.clear()
    pipeline.answer_prompt("ferrets", rag=True, judge=False, instance="none")
    assert "alpha-only sentinel" not in sent.get("user", "")


# ===========================================================================
# E1 — web/aggregator in-process settings functions
# ===========================================================================

def test_aiar_models_inprocess(monkeypatch):
    """get_models returns {active, default, models, source, ollama_reachable}
    via in-process aiar.llm (no proxy hop)."""
    from web import aggregator
    from aiar.llm import ollama_client
    monkeypatch.setenv("OLLAMA_MODEL", "qwen-test:1b")
    ollama_client.reset_for_testing()
    monkeypatch.setattr(
        ollama_client, "list_models",
        lambda *a, **k: [{"name": "qwen-test:1b", "size_bytes": 1, "family": "qwen"},
                         {"name": "qwen-big:9b", "size_bytes": 2, "family": "qwen"}])
    monkeypatch.setattr(ollama_client, "healthcheck", lambda: True)
    payload = aggregator.get_models()
    assert payload["active"] == "qwen-test:1b"
    assert payload["default"] == "qwen-test:1b"
    assert payload["ollama_reachable"] is True
    actives = [m for m in payload["models"] if m["active"]]
    assert len(actives) == 1 and actives[0]["name"] == "qwen-test:1b"
    ollama_client.reset_for_testing()


def test_aiar_models_inprocess_unreachable(monkeypatch):
    from web import aggregator
    from aiar.llm import ollama_client
    ollama_client.reset_for_testing()
    monkeypatch.setattr(ollama_client, "list_models", lambda *a, **k: [])
    monkeypatch.setattr(ollama_client, "healthcheck", lambda: False)
    payload = aggregator.get_models()
    assert payload["models"] == []
    assert payload["ollama_reachable"] is False
    ollama_client.reset_for_testing()


def test_aiar_models_inprocess_reachable_but_empty(monkeypatch):
    from web import aggregator
    from aiar.llm import ollama_client
    ollama_client.reset_for_testing()
    monkeypatch.setattr(ollama_client, "list_models", lambda *a, **k: [])
    monkeypatch.setattr(ollama_client, "healthcheck", lambda: True)
    payload = aggregator.get_models()
    assert payload["models"] == []
    assert payload["ollama_reachable"] is True
    ollama_client.reset_for_testing()


def test_aiar_set_active_inprocess(monkeypatch):
    from web import aggregator
    from aiar.llm import ollama_client
    monkeypatch.setenv("OLLAMA_MODEL", "qwen-test:1b")
    ollama_client.reset_for_testing()
    monkeypatch.setattr(
        ollama_client, "list_models",
        lambda *a, **k: [{"name": "qwen-test:1b"}, {"name": "qwen-big:9b"}])
    res = aggregator.set_active_model("qwen-big:9b")
    assert res["ok"] and res["data"]["active"] == "qwen-big:9b"
    assert res["data"]["previous"] == "qwen-test:1b"
    assert ollama_client.active_model() == "qwen-big:9b"
    # uninstalled -> 422
    bad = aggregator.set_active_model("nope:1b")
    assert not bad["ok"] and bad["status"] == 422
    ollama_client.reset_for_testing()


def test_aiar_rag_instances_inprocess(fresh_store):
    from web import aggregator
    fresh_store.create_instance("alpha", display_name="Alpha")
    payload = aggregator.get_rag_instances()
    names = {i["name"] for i in payload["instances"]}
    assert {"default", "alpha"} <= names
    assert payload["active"] == "default"
    assert payload["active_display_name"] == "Example RAG"
    by_name = {i["name"]: i for i in payload["instances"]}
    assert by_name["default"]["display_name"] == "Example RAG"
    # set active to alpha
    res = aggregator.set_active_rag("alpha")
    assert res["ok"] and res["data"]["active"] == "alpha"
    assert res["data"]["active_display_name"] == "Alpha"
    assert fresh_store.active_instance() == "alpha"
    # "none" is selectable
    res_none = aggregator.set_active_rag("none")
    assert res_none["ok"] and res_none["data"]["active"] == "none"
    assert res_none["data"]["active_display_name"] == "No RAG"


def test_aiar_delete_rag_inprocess(fresh_store):
    from web import aggregator
    fresh_store.create_instance("alpha", display_name="Alpha")
    # delete succeeds and reports the now-active instance
    res = aggregator.delete_rag_instance("alpha")
    assert res["ok"] and res["data"]["deleted"] == "alpha"
    assert res["data"]["active"] == "default"
    assert res["data"]["active_display_name"] == "Example RAG"
    names = {i["name"] for i in aggregator.get_rag_instances()["instances"]}
    assert "alpha" not in names
    # guards: default / none / unknown are rejected, never silently dropped
    assert aggregator.delete_rag_instance("default")["status"] == 422
    assert aggregator.delete_rag_instance("none")["status"] == 422
    assert aggregator.delete_rag_instance("")["status"] == 400
    assert aggregator.delete_rag_instance("ghost")["status"] == 422


def test_aiar_system_prompt_presets_inprocess(tmp_path, monkeypatch):
    """Named presets persist to <base>/system-prompts.json: save/list/overwrite,
    the 5-cap, delete, and not-found — independent of the active system prompt."""
    from web import aggregator
    monkeypatch.setenv("AIAR_BASE_DIR", str(tmp_path))

    assert aggregator.list_system_prompts()["presets"] == []

    res = aggregator.save_system_prompt_preset("Strict", "Be terse and literal.")
    assert res["ok"] and res["data"]["saved"] == "Strict"
    assert [p["name"] for p in aggregator.list_system_prompts()["presets"]] == ["Strict"]
    assert (tmp_path / "system-prompts.json").exists()

    # overwrite by same name keeps a single entry with updated text
    aggregator.save_system_prompt_preset("Strict", "Updated text.")
    presets = aggregator.list_system_prompts()["presets"]
    assert len(presets) == 1 and presets[0]["text"] == "Updated text."

    # fill to the cap of 5, then a 6th new name is rejected
    for i in range(4):
        assert aggregator.save_system_prompt_preset(f"P{i}", "x")["ok"]
    assert len(aggregator.list_system_prompts()["presets"]) == 5
    over = aggregator.save_system_prompt_preset("P5", "x")
    assert not over["ok"] and over["status"] == 422 and over["error"] == "preset_limit"
    # but overwriting an existing one still works at the cap
    assert aggregator.save_system_prompt_preset("Strict", "y")["ok"]

    # validation: empty name / empty prompt
    assert aggregator.save_system_prompt_preset("", "x")["status"] == 400
    assert aggregator.save_system_prompt_preset("ok", "   ")["status"] == 422

    # delete
    d = aggregator.delete_system_prompt_preset("Strict")
    assert d["ok"] and "Strict" not in [p["name"] for p in d["data"]["presets"]]
    assert aggregator.delete_system_prompt_preset("nope")["status"] == 404


def test_aiar_system_prompt_inprocess():
    from web import aggregator
    from aiar.harness import pipeline
    pipeline.reset_system_prompt()
    cur = aggregator.get_system_prompt()
    assert cur["text"] == pipeline.ANSWER_SYSTEM_PROMPT
    assert cur["source"] == "default"
    res = aggregator.set_system_prompt("MY CUSTOM PROMPT")
    assert res["ok"] and res["data"]["source"] == "active"
    assert aggregator.get_system_prompt()["text"] == "MY CUSTOM PROMPT"
    # empty resets to default
    aggregator.set_system_prompt("")
    assert aggregator.get_system_prompt()["source"] == "default"
    pipeline.reset_system_prompt()


# ===========================================================================
# E2 — web/server routes for the six /api/* settings paths
# ===========================================================================

def _make_handler():
    """Build a WatcherHandler without binding a socket, for direct method
    testing of the GET/POST ladders (no live server needed)."""
    import io
    from web import server as web_server

    class _FakeHandler(web_server.WatcherHandler):
        def __init__(self):  # bypass BaseHTTPRequestHandler.__init__
            self._captured = {}
            self.wfile = io.BytesIO()
            self.headers = {}

        def send_response(self, code, *a):
            self._captured["status"] = int(code)

        def send_header(self, *a, **k):
            pass

        def end_headers(self):
            pass

    return _FakeHandler


def test_aiar_settings_routes(fresh_store, monkeypatch):
    """The server GET/POST ladders serve the settings page + the six APIs."""
    import json as _json
    from web import server as web_server
    from aiar.llm import ollama_client
    Handler = _make_handler()
    ollama_client.reset_for_testing()
    monkeypatch.setattr(ollama_client, "list_models",
                        lambda *a, **k: [{"name": ollama_client.active_model()}])

    # static page registered
    assert "/settings" in web_server._STATIC
    assert "/settings.js" in web_server._STATIC

    # GET /api/models
    h = Handler()
    h.path = "/api/models"
    h.do_GET()
    assert h._captured["status"] == 200
    body = _json.loads(h.wfile.getvalue().decode())
    assert "models" in body and "active" in body

    # GET /api/rag/instances
    h = Handler()
    h.path = "/api/rag/instances"
    h.do_GET()
    body = _json.loads(h.wfile.getvalue().decode())
    assert "instances" in body and any(i["name"] == "default" for i in body["instances"])

    # GET /api/system-prompt
    h = Handler()
    h.path = "/api/system-prompt"
    h.do_GET()
    body = _json.loads(h.wfile.getvalue().decode())
    assert "text" in body and "source" in body

    ollama_client.reset_for_testing()


# ===========================================================================
# E3 — canonical settings.html / settings.js parity
# ===========================================================================

def test_aiar_settings_parity():
    """The settings page exists with three cards and consumes the shared
    /api/... contract (model, rag incl 'No RAG', system-prompt)."""
    from pathlib import Path
    static = Path(__file__).resolve().parents[1] / "web" / "static"
    html = (static / "settings.html").read_text()
    js = (static / "settings.js").read_text()
    # three cards
    assert "settings-model-select" in html
    assert "settings-rag-select" in html
    assert "settings-system-prompt" in html
    # No RAG first-class option present in the rendering logic
    assert "No RAG" in js or "No RAG" in html
    # consumes the shared API paths
    for path in ("/api/models", "/api/rag/instances", "/api/system-prompt"):
        assert path in js
    # system-prompt presets: dropdown + save + delete, wired to the presets API
    assert "system-preset-select" in html
    assert "save-preset" in html and "delete-preset" in html
    for path in ("/api/system-prompts", "/api/system-prompts/save", "/api/system-prompts/delete"):
        assert path in js
    # escapeHtml + no-store idiom (matches the AIAR static page idiom)
    assert "escapeHtml" in js and "no-store" in js


def test_aiar_simulate_page_parity():
    """Simulate page: 'Prompt Console' heading, no stale description sentence,
    and an LLM-judging toggle wired into the simulate payload."""
    from pathlib import Path
    static = Path(__file__).resolve().parents[1] / "web" / "static"
    html = (static / "index.html").read_text()
    js = (static / "app.js").read_text()
    assert "Prompt Console" in html
    assert "answered by your Qwen model, then judged" not in html
    assert 'id="judge"' in html
    assert '$("judge").checked' in js


def test_aiar_activity_clear_parity():
    """The Activity page has a Clear Recent Activity control wired to the API."""
    from pathlib import Path
    static = Path(__file__).resolve().parents[1] / "web" / "static"
    assert "clear-activity" in (static / "activity.html").read_text()
    assert "/api/activity/clear" in (static / "activity.js").read_text()


def test_aiar_titles_title_cased():
    """GUI headings/labels use Title Case (regression guard for the casing pass)."""
    from pathlib import Path
    static = Path(__file__).resolve().parents[1] / "web" / "static"
    activity = (static / "activity.html").read_text()
    settings = (static / "settings.html").read_text()
    assert "Recent Activity" in activity and "Recent activity" not in activity
    assert "Evaluation Queue" in (static / "evaluation.html").read_text()
    for label in ("RAG Instance", "System Prompt", "Active Model", "Reset to Default"):
        assert label in settings


def test_aiar_feature_guides_present():
    """Retrieval-quality feature guides exist and are surfaced by the prompt/docs."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    guides = root / "examples" / "feature-guides"
    for name in ("README.md", "reranker.md", "hybrid-retrieval.md",
                 "query-rewrite-hyde.md", "grounding-reinjection.md", "top-k.md",
                 "measure-lift.md", "improving-rag.md"):
        assert (guides / name).exists(), f"missing feature guide {name}"
    readme = (root / "README.md").read_text()
    assert "NEXT STEPS" in readme and "examples/feature-guides/" in readme
    assert "examples/feature-guides/" in (root / "PLAYBOOK.md").read_text()


def test_aiar_nav_pill_added_all_pages():
    from pathlib import Path
    static = Path(__file__).resolve().parents[1] / "web" / "static"
    for page in ("index.html", "activity.html", "evaluation.html"):
        assert "/settings" in (static / page).read_text(), f"{page} missing Settings nav pill"


def test_aiar_fusion_get_by_ids_scoped_to_instance(fresh_store):
    """get_by_ids materialises only from the named instance's collection."""
    fresh_store.create_instance("alpha")
    fresh_store.create_instance("beta")
    fresh_store.add([_chunk("a.md", "alpha unique sentinel text", idx=0)],
                    instance="alpha")
    fresh_store.add([_chunk("b.md", "beta unique sentinel text", idx=0)],
                    instance="beta")
    a_ids, _ = fresh_store.all_documents(instance="alpha")
    assert a_ids
    # Asking beta for alpha's id returns nothing — no cross-instance read.
    got = fresh_store.get_by_ids(a_ids, instance="beta")
    assert got == []
    got_a = fresh_store.get_by_ids(a_ids, instance="alpha")
    assert got_a and got_a[0].id == a_ids[0]
