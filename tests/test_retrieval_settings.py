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
