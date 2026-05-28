"""Tests for the runtime retrieval-feature overrides layer + API + wiring."""
from __future__ import annotations

import pytest

from aiar.rag import settings as rs


def setup_function():
    rs.reset()


def teardown_function():
    rs.reset()


def test_default_and_env(monkeypatch):
    monkeypatch.delenv("RAG_HYBRID_ENABLED", raising=False)
    assert rs.get("hybrid") is False
    assert rs.source("hybrid") == "default"
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "1")
    assert rs.get("hybrid") is True
    assert rs.source("hybrid") == "env"


def test_override_precedence_and_reset(monkeypatch):
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "1")
    rs.set_override("hybrid", False)
    assert rs.get("hybrid") is False and rs.source("hybrid") == "override"
    rs.reset()
    assert rs.get("hybrid") is True  # back to env


def test_validation():
    rs.set_override("rewrite_mode", "hyde")
    assert rs.get("rewrite_mode") == "hyde"
    with pytest.raises(ValueError):
        rs.set_override("rewrite_mode", "bogus")
    with pytest.raises(ValueError):
        rs.set_override("top_k", 0)
    with pytest.raises(ValueError):
        rs.set_override("top_k", "abc")
    with pytest.raises(ValueError):
        rs.set_override("does_not_exist", 1)


def test_effective_shape():
    cfg = rs.effective()
    for k in ("hybrid", "rerank", "fetch_k", "top_k", "rewrite_mode",
              "grounding_reinjection", "rerank_model"):
        assert k in cfg


def test_retriever_honors_overrides():
    from aiar.rag import retriever
    assert retriever.hybrid_enabled() is False
    rs.set_override("hybrid", True)
    rs.set_override("rerank", True)
    assert retriever.hybrid_enabled() is True
    assert retriever.rerank_enabled() is True


def test_grounding_and_rewrite_honor_overrides():
    from aiar.grounding.reinject import reinjection_enabled
    from aiar.rag import query_rewrite
    assert reinjection_enabled() is False
    rs.set_override("grounding_reinjection", True)
    assert reinjection_enabled() is True
    rs.set_override("rewrite_mode", "hyde")
    assert query_rewrite._mode() == "hyde"


def test_aggregator_get_set_reset():
    from web import aggregator
    got = aggregator.get_retrieval_settings()
    assert {"config", "sources", "defaults"} <= set(got)
    res = aggregator.set_retrieval_setting("hybrid", True)
    assert res["ok"] and res["data"]["config"]["hybrid"] is True
    bad_key = aggregator.set_retrieval_setting("nope", True)
    assert not bad_key["ok"] and bad_key["status"] == 400
    bad_val = aggregator.set_retrieval_setting("rewrite_mode", "bogus")
    assert not bad_val["ok"] and bad_val["status"] == 422
    reset = aggregator.reset_retrieval_settings()
    assert reset["ok"] and reset["data"]["config"]["hybrid"] is False


def test_settings_page_parity():
    from pathlib import Path
    static = Path(__file__).resolve().parents[1] / "web" / "static"
    html = (static / "settings.html").read_text()
    js = (static / "settings.js").read_text()
    for el in ("rf-hybrid", "rf-rerank", "rf-grounding", "rf-rewrite",
               "rf-topk", "rf-fetchk", "rf-reset"):
        assert el in html, f"settings.html missing {el}"
    assert "/api/retrieval" in js
    assert "retrieval-badges" in (static / "index.html").read_text()
    assert "renderRetrievalBadges" in (static / "app.js").read_text()


def test_simulate_page_grounding_dropdowns_present():
    """The Simulate page's Grounding Context section uses save-on-change
    dropdowns wired to the existing /api/* endpoints."""
    from pathlib import Path
    static = Path(__file__).resolve().parents[1] / "web" / "static"
    html = (static / "index.html").read_text()
    js = (static / "app.js").read_text()
    # One dropdown per field, each with a status span next to it.
    for el in ("gc-model-select", "gc-corpus-select", "gc-system-prompt-select",
               "gc-feat-hybrid", "gc-feat-rerank", "gc-feat-rewrite_mode",
               "gc-feat-grounding_reinjection", "gc-feat-top_k", "gc-feat-fetch_k"):
        assert el in html, f"index.html missing {el}"
    for el in ("gc-model-status", "gc-corpus-status", "gc-system-prompt-status",
               "gc-feat-hybrid-status", "gc-feat-rerank-status",
               "gc-feat-rewrite_mode-status",
               "gc-feat-grounding_reinjection-status",
               "gc-feat-top_k-status", "gc-feat-fetch_k-status"):
        assert el in html, f"index.html missing status span {el}"
    # Save-on-change wiring: each select fires a POST on the 'change' event.
    assert 'addEventListener("change", onModelChange)' in js
    assert 'addEventListener("change", onCorpusChange)' in js
    assert 'addEventListener("change", onSystemPromptChange)' in js
    assert 'addEventListener("change", onRetrievalFeatureChange)' in js
    # The page hits the existing setter endpoints — no new ones invented.
    for endpoint in ("/api/models/active", "/api/rag/active",
                     "/api/system-prompt", "/api/retrieval"):
        assert endpoint in js, f"app.js missing {endpoint}"
    # No save buttons in the Grounding Context section — save-on-change only.
    assert "<button id=\"run\"" in html  # the prompt 'Run' button still exists
    # The grounding context section itself has no button child.
    start = html.index("Grounding Context")
    end = html.index("Prompt Console")
    section = html[start:end]
    assert "<button" not in section, "Grounding Context must not have a save button"


def test_accept_judge_button_wired_to_existing_verdict_endpoint():
    """The Simulate page's 'Accept LLM Judge Evaluation' button reuses the
    existing /api/evaluation/verdict endpoint with the judge's rating mapped
    to a score and the judge's reason as the correction. No new backend
    surface is introduced."""
    from pathlib import Path
    static = Path(__file__).resolve().parents[1] / "web" / "static"
    html = (static / "index.html").read_text()
    js = (static / "app.js").read_text()

    # The button exists in the result panel's eval-actions row, starts hidden,
    # and lives next to the existing 'Mark for Evaluation' button.
    assert 'id="accept-judge"' in html
    assert 'Accept LLM Judge Evaluation' in html
    # The accept button appears AFTER mark-for-evaluation in the same actions
    # row — so users see "mark, then accept" reading left to right.
    mark_idx = html.index('id="mark"')
    accept_idx = html.index('id="accept-judge"')
    assert mark_idx < accept_idx, "Accept button must follow Mark button"

    # Click handler is wired and POSTs to the existing verdict endpoint with
    # a rating-mapped score and the judge's reason as correction.
    assert 'acceptJudgeEvaluation' in js
    assert '"/api/evaluation/verdict"' in js
    assert '_ratingToScore' in js
    # The mapping preserves the judge's rating round-trip via _score_to_rating:
    # good>=8, partial>=4, bad<4.
    assert 'if (rating === "good") return 8' in js
    assert 'if (rating === "partial") return 5' in js

    # The button is suppressed when the judge could not produce a usable
    # reason (timeout, unparseable, etc.) — those failure tags must not be
    # silently grounded.
    for tag in ("judge_failed", "judge_timeout", "judge_unparseable"):
        assert tag in js, f"app.js must guard on {tag}"


def test_judge_timeout_default_is_180s():
    """The judge's default per-call timeout matches the rest of the framework
    (180s) so a slow local model on the answer path doesn't time out only the
    judge call."""
    import importlib, os
    os.environ.pop("EVAL_JUDGE_TIMEOUT_S", None)
    from aiar.eval import judge as _judge
    importlib.reload(_judge)
    assert _judge._JUDGE_TIMEOUT_S == 180


def test_aggregator_setters_match_dropdown_contracts(monkeypatch):
    """The endpoints the Simulate-page dropdowns POST to all return the
    {ok, data} envelope that app.js expects."""
    from web import aggregator
    # Retrieval feature save round-trips.
    res = aggregator.set_retrieval_setting("hybrid", True)
    assert res["ok"] and "config" in res["data"]
    res = aggregator.set_retrieval_setting("rewrite_mode", "hyde")
    assert res["ok"] and res["data"]["config"]["rewrite_mode"] == "hyde"
    res = aggregator.set_retrieval_setting("top_k", 5)
    assert res["ok"] and res["data"]["config"]["top_k"] == 5
    # System-prompt save round-trip (reset path).
    res = aggregator.set_system_prompt("")
    assert res["ok"] and res["data"]["source"] == "default"
    res = aggregator.set_system_prompt("Be terse.")
    assert res["ok"] and res["data"]["text"] == "Be terse."
    # Reset back to default so we don't pollute other tests.
    aggregator.set_system_prompt("")
