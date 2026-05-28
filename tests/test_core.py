"""Smoke + unit tests for the AIAR core that need NO live model or chromadb.

Run:  python -m pytest tests/ -q
"""
from __future__ import annotations

from pathlib import Path

from aiar import doctor
from aiar.rag.ingest import ingest_folder, ingest_file, Chunk
from aiar.eval.scorer import score_answer
from aiar.eval.judge import judge_answer
from aiar.eval.schemas import Verdict
from aiar.harness import pipeline
from aiar.grounding import store as grounding_store
from aiar.grounding.reinject import grounding_block

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "docs"


# ---- ingest ----------------------------------------------------------------

def test_ingest_folder_produces_chunks():
    chunks = ingest_folder(EXAMPLES)
    assert chunks, "expected at least one chunk from the example docs"
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("30 day" in c.text.lower() for c in chunks)


def test_ingest_json(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"name": "Acme", "window_days": 30}', encoding="utf-8")
    chunks = ingest_file(p)
    assert chunks
    assert "30" in chunks[0].text


def test_ingest_file_strips_front_matter_and_keeps_metadata(tmp_path):
    p = tmp_path / "record.md"
    p.write_text(
        "---\n"
        "claim_type: guideline\n"
        "authority_level: 1\n"
        "condition: insomnia\n"
        "---\n\n"
        "# Body\n\n"
        "Melatonin is not strongly recommended for chronic insomnia.\n",
        encoding="utf-8",
    )
    chunks = ingest_file(p)
    assert chunks
    assert "claim_type:" not in chunks[0].text
    assert chunks[0].metadata["claim_type"] == "guideline"
    assert chunks[0].metadata["authority_level"] == 1


# ---- deterministic scorer --------------------------------------------------

def test_scorer_match_and_forbid():
    rubric = [
        {"id": "has-30", "weight": 2, "match_any": ["30 day"]},
        {"id": "no-bad", "weight": 1, "forbid_any": ["lifetime"]},
    ]
    good = score_answer("You have 30 days for a refund.", rubric)
    assert good.score == 3 and good.passed
    bad = score_answer("Refunds last a lifetime, no limit.", rubric)
    assert bad.score == 0 and not bad.passed


# ---- judge (mocked LLM) ----------------------------------------------------

def test_judge_parses_verdict_with_mock():
    def fake_caller(system, user, **kwargs):
        return ('{"rating":"good","reason":"correct","failure_tags":[],"confidence":"high"}', 5)
    v = judge_answer("q", "a", "ctx", llm_caller=fake_caller)
    assert isinstance(v, Verdict)
    assert v.rating == "good" and v.confidence == "high"


def test_judge_handles_garbage_with_mock():
    def fake_caller(system, user, **kwargs):
        return ("not json at all", 5)
    v = judge_answer("q", "a", llm_caller=fake_caller)
    assert v.rating == "partial" and "judge_unparseable" in v.failure_tags


# ---- grounding store + reinjection -----------------------------------------

def test_grounding_record_and_lookup(tmp_path):
    v = Verdict(rating="bad", reason="said 90 days", failure_tags=["wrong_number"])
    grounding_store.record("How long is the refund window?", v,
                           correction="The refund window is 30 days.", base=tmp_path)
    found = grounding_store.lookup("how long is the REFUND window?!", base=tmp_path)
    assert found and found[-1].correction == "The refund window is 30 days."


def test_grounding_block_renders_correction(tmp_path, monkeypatch):
    monkeypatch.setenv("GROUNDING_BASE_DIR", str(tmp_path))
    grounding_store.reset_default_store_for_testing()
    v = Verdict(rating="bad", reason="wrong", failure_tags=["x"])
    grounding_store.record("test prompt", v, correction="the right answer")
    block = grounding_block("test prompt", force=True)
    assert "the right answer" in block
    grounding_store.reset_default_store_for_testing()


def test_answer_prompt_passes_through_to_llm_when_rag_has_no_context(monkeypatch):
    """When RAG is on but retrieval came back empty, the answerer routes the
    prompt to the LLM with a '[No relevant evidence retrieved from the corpus.]'
    note instead of returning a canned refusal, so the model can decline
    domain-specific questions but still answer arithmetic / general knowledge
    per ANSWER_SYSTEM_PROMPT."""
    seen = {}

    def fake_call(system_prompt, user_prompt, **kwargs):
        seen["user"] = user_prompt
        return "the answer", 42

    monkeypatch.setattr("aiar.rag.retriever.get_context", lambda *args, **kwargs: "")
    monkeypatch.setattr("aiar.grounding.reinject.grounding_block", lambda *args, **kwargs: "")
    monkeypatch.setattr("aiar.grounding.reinject.reinjection_enabled", lambda: False)
    monkeypatch.setattr("aiar.harness.pipeline.call_ollama", fake_call)
    monkeypatch.setattr("aiar.eval.judge.judge_answer",
                        lambda *args, **kwargs: Verdict(
                            rating="good", reason="ok",
                            failure_tags=[], confidence="high"))
    out = pipeline.answer_prompt("what is 2+2?", rag=True, judge=False)
    assert "[No relevant evidence retrieved from the corpus.]" in seen["user"]
    assert out["answer"] == "the answer"
    assert out["grounded"] is False


# ---- answer_prompt kwargs: retrieval_query / retrieval_where / rewrite ------

def _stub_llm_and_grounding(monkeypatch, response="ok"):
    """Stub the LLM + grounding helpers to keep these tests offline + fast."""
    monkeypatch.setattr("aiar.harness.pipeline.call_ollama",
                        lambda *a, **k: (response, 5))
    monkeypatch.setattr("aiar.grounding.reinject.grounding_block",
                        lambda *a, **k: "")
    monkeypatch.setattr("aiar.grounding.reinject.reinjection_enabled",
                        lambda: False)
    monkeypatch.setattr("aiar.eval.judge.judge_answer",
                        lambda *a, **k: Verdict(
                            rating="good", reason="ok",
                            failure_tags=[], confidence="high"))


def test_answer_prompt_retrieval_query_overrides_prompt_text(monkeypatch):
    """``retrieval_query=`` is what's sent to the retriever, leaving the
    answerer's user prompt as the original question."""
    seen = {}

    def fake_get_context(query, *, top_k=None, instance=None, where=None, rewrite=True):
        seen["query"] = query
        return ""

    monkeypatch.setattr("aiar.rag.retriever.get_context", fake_get_context)
    _stub_llm_and_grounding(monkeypatch)
    pipeline.answer_prompt(
        "answerer-prompt",
        rag=True,
        judge=False,
        retrieval_query="custom-retrieval-query",
    )
    assert seen["query"] == "custom-retrieval-query"


def test_answer_prompt_retrieval_query_defaults_to_prompt(monkeypatch):
    """Without ``retrieval_query=``, the retriever is queried with the prompt
    itself — the historical default behavior is preserved."""
    seen = {}
    monkeypatch.setattr("aiar.rag.retriever.get_context",
                        lambda q, **kw: seen.update(query=q) or "")
    _stub_llm_and_grounding(monkeypatch)
    pipeline.answer_prompt("the original prompt", rag=True, judge=False)
    assert seen["query"] == "the original prompt"


def test_answer_prompt_retrieval_where_passed_to_retriever(monkeypatch):
    """``retrieval_where=`` is forwarded as the Chroma metadata filter."""
    seen = {}
    monkeypatch.setattr("aiar.rag.retriever.get_context",
                        lambda q, **kw: seen.update(where=kw.get("where")) or "")
    _stub_llm_and_grounding(monkeypatch)
    pipeline.answer_prompt(
        "prompt",
        rag=True,
        judge=False,
        retrieval_where={"source": "wiki"},
    )
    assert seen["where"] == {"source": "wiki"}


def test_answer_prompt_rewrite_flag_passed_to_retriever(monkeypatch):
    """``rewrite=False`` propagates and disables HyDE / query-rewrite paths."""
    seen = {}
    monkeypatch.setattr("aiar.rag.retriever.get_context",
                        lambda q, **kw: seen.update(rewrite=kw.get("rewrite")) or "")
    _stub_llm_and_grounding(monkeypatch)
    pipeline.answer_prompt("prompt", rag=True, judge=False, rewrite=False)
    assert seen["rewrite"] is False
    # Default is True when not specified.
    seen.clear()
    pipeline.answer_prompt("prompt", rag=True, judge=False)
    assert seen["rewrite"] is True


def test_retriever_uses_vector_only_path_when_where_is_set(monkeypatch):
    """When a ``where=`` metadata filter is supplied, ``_retrieve_candidates``
    must skip the BM25/hybrid path — Chroma metadata filters can't be enforced
    against the text-only BM25 index, so a hybrid fuse would dilute the filter.
    """
    from aiar.rag import retriever as _retriever
    hybrid_called = {"n": 0}

    def fake_hybrid_enabled():
        hybrid_called["n"] += 1
        return True

    def fake_query_scored(query, n, *, where=None, instance=None):
        return [("chunk", 0.9, {"meta": where})]

    monkeypatch.setattr(_retriever, "hybrid_enabled", fake_hybrid_enabled)
    monkeypatch.setattr("aiar.rag.store.query_scored", fake_query_scored)
    result = _retriever._retrieve_candidates(
        "q", 5, where={"source": "wiki"}, instance=None,
    )
    # Vector-only path was taken: hybrid_enabled() should NOT have been
    # consulted because the where-shortcut returns first.
    assert hybrid_called["n"] == 0
    assert result == [("chunk", 0.9, {"meta": {"source": "wiki"}})]


# ---- judge_criteria ---------------------------------------------------------

def test_judge_answer_includes_criteria_in_user_prompt():
    """``criteria=`` is rendered into the judge's user prompt under an
    EVAL TARGET heading so the judge model knows what rubric to apply."""
    captured = {}

    def fake_caller(system, user, **kw):
        captured["user"] = user
        return ('{"rating":"good","reason":"ok","failure_tags":[],"confidence":"high"}', 7)

    judge_answer(
        "prompt-text",
        "response-text",
        context="ctx",
        criteria="must reference at least one statute",
        llm_caller=fake_caller,
    )
    assert "EVAL TARGET:" in captured["user"]
    assert "must reference at least one statute" in captured["user"]


def test_judge_answer_omits_eval_target_when_no_criteria():
    """When ``criteria=`` is empty, the EVAL TARGET section must not appear —
    no surprise extra content for callers that don't opt in."""
    captured = {}

    def fake_caller(system, user, **kw):
        captured["user"] = user
        return ('{"rating":"good","reason":"ok","failure_tags":[],"confidence":"high"}', 7)

    judge_answer(
        "prompt-text",
        "response-text",
        context="ctx",
        llm_caller=fake_caller,
    )
    assert "EVAL TARGET:" not in captured["user"]


def test_answer_prompt_forwards_judge_criteria_to_judge(monkeypatch):
    """``judge_criteria=`` on ``answer_prompt`` reaches the judge as ``criteria=``."""
    seen = {}

    def fake_judge(prompt, response, context, *, criteria="", llm_caller=None):
        seen["criteria"] = criteria
        return Verdict(rating="good", reason="ok",
                       failure_tags=[], confidence="high")

    monkeypatch.setattr("aiar.eval.judge.judge_answer", fake_judge)
    monkeypatch.setattr("aiar.rag.retriever.get_context", lambda *a, **k: "")
    monkeypatch.setattr("aiar.grounding.reinject.grounding_block",
                        lambda *a, **k: "")
    monkeypatch.setattr("aiar.grounding.reinject.reinjection_enabled", lambda: False)
    monkeypatch.setattr("aiar.harness.pipeline.call_ollama",
                        lambda *a, **k: ("answer", 5))
    pipeline.answer_prompt(
        "prompt",
        rag=False,
        judge=True,
        judge_criteria="be concise",
    )
    assert seen["criteria"] == "be concise"


def test_doctor_reports_healthy_install(monkeypatch):
    monkeypatch.setattr(doctor, "_module_present", lambda name: True)
    monkeypatch.setattr("aiar.llm.healthcheck", lambda: True)
    monkeypatch.setattr("aiar.llm.active_model", lambda: "qwen2.5:7b")
    monkeypatch.setattr("aiar.llm.list_models",
                        lambda show_all=False: [{"name": "qwen2.5:7b"}])
    monkeypatch.setattr("aiar.rag.store.init", lambda: None)
    monkeypatch.setattr("aiar.rag.store.is_ready", lambda: True)

    report = doctor.run_checks()
    assert report["overall"] == "pass"
    assert any(c["name"] == "ollama" and c["level"] == "pass" for c in report["checks"])
    assert any(c["name"] == "rag_store" and c["level"] == "pass" for c in report["checks"])


def test_doctor_fails_when_ollama_and_rag_deps_missing(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_module_present",
        lambda name: name not in {"chromadb", "sentence_transformers", "rank_bm25"},
    )
    monkeypatch.setattr("aiar.llm.healthcheck", lambda: False)
    monkeypatch.setattr("aiar.llm.active_model", lambda: "qwen2.5:7b")
    monkeypatch.setattr("aiar.llm.list_models", lambda show_all=False: [])

    report = doctor.run_checks()
    assert report["overall"] == "fail"
    assert any(c["name"] == "rag_deps" and c["level"] == "fail" for c in report["checks"])
    assert any(c["name"] == "ollama" and c["level"] == "fail" for c in report["checks"])
