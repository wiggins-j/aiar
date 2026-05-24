"""Smoke + unit tests for the AIAR core that need NO live model or chromadb.

Run:  python -m pytest tests/ -q
"""
from __future__ import annotations

from pathlib import Path

from aiar.rag.ingest import ingest_folder, ingest_file, Chunk
from aiar.eval.scorer import score_answer
from aiar.eval.judge import judge_answer
from aiar.eval.schemas import Verdict
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
    assert v.rating == "bad" and "judge_unparseable" in v.failure_tags


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
