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
