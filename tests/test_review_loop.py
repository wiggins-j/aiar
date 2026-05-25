from __future__ import annotations

from pathlib import Path

from aiar.grounding import store as grounding_store
from aiar.observability import observer
from web import aggregator
from web.config import Config


def _config(tmp_path: Path) -> Config:
    return Config(
        static_dir=tmp_path,
        host="127.0.0.1",
        port=8088,
        queue_file=tmp_path / "eval-queue.jsonl",
        verdicts_file=tmp_path / "verdicts.jsonl",
        reason_threshold=7,
    )


def test_answer_prompt_uses_captured_call_id(monkeypatch):
    from aiar.harness import pipeline
    from aiar.rag import retriever

    def fake_call(system, user, **kwargs):
        capture = kwargs.get("capture")
        if capture is not None:
            capture["call_id"] = "call-123"
        return ("answer text", 7)

    monkeypatch.setattr(pipeline, "call_ollama", fake_call)
    monkeypatch.setattr(retriever, "get_context", lambda *args, **kwargs: "")

    result = pipeline.answer_prompt("test prompt", judge=False)
    assert result["call_id"] == "call-123"


def test_clear_recent_activity(tmp_path, monkeypatch):
    """clear_recent_activity() wipes the observer call log; verdicts/queue (which
    keep their own copies) are unaffected."""
    from aiar.observability import observer
    from web import aggregator
    monkeypatch.setenv("AIAR_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("AIAR_OBSERVER_ENABLED", raising=False)
    for i in range(2):
        token = observer.set_context(endpoint="/test", raw_prompt=f"q{i}")
        try:
            observer.emit_call(model="m", system_prompt="s", user_prompt="u",
                               options={}, format=None, think=False,
                               response_text="a", thinking=None, prompt_tokens=1,
                               completion_tokens=1, latency_ms=1, error=None)
        finally:
            observer.clear_context(token)
    assert len(observer.read_recent(50)) == 2
    res = aggregator.clear_recent_activity()
    assert res["ok"] and res["data"]["cleared"] == 2
    assert observer.read_recent(50) == []


def test_answer_prompt_includes_retrieval_block(monkeypatch):
    """answer_prompt reports the retrieval features that were in effect."""
    from aiar.harness import pipeline
    from aiar.rag import retriever, settings as rs
    rs.reset()
    monkeypatch.setattr(pipeline, "call_ollama", lambda s, u, **k: ("ans", 5))
    monkeypatch.setattr(retriever, "get_context", lambda *a, **k: "")
    rs.set_override("hybrid", True)
    try:
        res = pipeline.answer_prompt("q", rag=True, judge=False, top_k=4)
        r = res["retrieval"]
        assert r["rag"] is True and r["top_k"] == 4 and r["hybrid"] is True
        for k in ("rerank", "rewrite_mode", "grounding_reinjection"):
            assert k in r
    finally:
        rs.reset()


def test_simulate_forwards_judge_toggle(monkeypatch):
    """The Simulate page's LLM-judging toggle flows through to answer_prompt:
    judge=False skips the LLM-as-judge (no verdict)."""
    import aiar.harness as harness
    captured = {}

    def fake_answer(prompt, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return {"answer": "a", "verdict": None, "call_id": "x"}

    monkeypatch.setattr(harness, "answer_prompt", fake_answer)
    aggregator.simulate_prompt("q", rag=False, judge=False)
    assert captured.get("judge") is False
    aggregator.simulate_prompt("q", rag=False, judge=True)
    assert captured.get("judge") is True


def test_queue_and_reground_use_raw_prompt(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    base_dir = tmp_path / "grounding"
    monkeypatch.setenv("AIAR_LOG_DIR", str(log_dir))
    monkeypatch.setenv("GROUNDING_BASE_DIR", str(base_dir))
    grounding_store.reset_default_store_for_testing()

    raw_prompt = "How long is the refund window?"
    llm_prompt = (
        "--- Knowledge (top-1) ---\n"
        "[chunk 1]\nRefunds are allowed within 30 days.\n"
        "--- End Knowledge ---\n\n"
        f"QUESTION:\n{raw_prompt}"
    )
    token = observer.set_context(endpoint="/eval/prompt", raw_prompt=raw_prompt)
    try:
        call_id = observer.emit_call(
            model="qwen-test:1b",
            system_prompt="system",
            user_prompt=llm_prompt,
            options={},
            format=None,
            think=False,
            response_text="Refunds are allowed within 30 days.",
            thinking=None,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=12,
            error=None,
        )
    finally:
        observer.clear_context(token)

    assert call_id
    config = _config(tmp_path)
    assert aggregator.enqueue(config, call_id)["ok"] is True

    queue = aggregator.evaluation_queue(config)
    assert queue["items"][0]["prompt"] == raw_prompt

    result = aggregator.submit_verdict(
        config, call_id, 3, correction="The refund window is 30 days.")
    assert result["ok"] is True

    found = grounding_store.lookup(raw_prompt, base=base_dir)
    assert found and found[-1].correction == "The refund window is 30 days."
    assert grounding_store.lookup(llm_prompt, base=base_dir) == []

    grounding_store.reset_default_store_for_testing()


def test_clear_evaluation_queue(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("AIAR_LOG_DIR", str(log_dir))

    raw_prompt = "hello there"
    token = observer.set_context(endpoint="/eval/prompt", raw_prompt=raw_prompt)
    try:
        call_id = observer.emit_call(
            model="qwen-test:1b",
            system_prompt="system",
            user_prompt="QUESTION:\nhello there",
            options={},
            format=None,
            think=False,
            response_text="Hello!",
            thinking=None,
            prompt_tokens=4,
            completion_tokens=2,
            latency_ms=8,
            error=None,
        )
    finally:
        observer.clear_context(token)

    config = _config(tmp_path)
    assert aggregator.enqueue(config, call_id)["ok"] is True
    assert aggregator.evaluation_queue(config)["count"] == 1

    cleared = aggregator.clear_evaluation_queue(config)
    assert cleared["ok"] is True
    assert cleared["data"]["cleared"] == 1
    assert aggregator.evaluation_queue(config)["count"] == 0


def test_recent_activity_and_detail_expose_previews_and_rag_state(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("AIAR_LOG_DIR", str(log_dir))

    raw_prompt = "Is live chat support available on weekends?"
    user_prompt = (
        "--- Knowledge (top-1) ---\n"
        "[chunk 1]\nLive chat is weekdays only.\n"
        "--- End Knowledge ---\n\n"
        f"QUESTION:\n{raw_prompt}"
    )
    token = observer.set_context(endpoint="/eval/prompt", raw_prompt=raw_prompt)
    try:
        call_id = observer.emit_call(
            model="qwen-test:1b",
            system_prompt="system",
            user_prompt=user_prompt,
            options={},
            format=None,
            think=False,
            response_text="No, live chat support is not available on weekends.",
            thinking=None,
            prompt_tokens=12,
            completion_tokens=8,
            latency_ms=15,
            error=None,
        )
    finally:
        observer.clear_context(token)

    config = _config(tmp_path)
    recent = aggregator.recent_activity(config, limit=5)
    item = recent["items"][0]
    assert item["prompt_preview"].startswith("Is live chat support available")
    assert item["response_preview"].startswith("No, live chat support is not available")
    assert item["rag_state"] == "RAG ON"

    detail = aggregator.activity_detail(config, call_id)
    assert detail["found"] is True
    assert detail["prompt"] == raw_prompt
    assert detail["response"] == "No, live chat support is not available on weekends."
    assert detail["rag_state"] == "RAG ON"
