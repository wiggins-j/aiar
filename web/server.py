"""Stdlib http.server for the AIAR watcher GUI.

Pages:
    GET  /                       -> simulate console (run a prompt)
    GET  /activity               -> recent LLM calls (mark for evaluation)
    GET  /evaluation             -> evaluation queue (score 1-10 + reground)
    GET  /settings               -> model / RAG instance / system-prompt settings

JSON API:
    POST /api/simulate           {prompt, rag?, think?, reground?, judge?, instance?}
    GET  /api/activity
    GET  /api/activity/detail?call_id=...
    POST /api/activity/evaluate  {call_id}                  (enqueue)
    POST /api/activity/clear                                (clear the call log)
    GET  /api/evaluation/queue
    POST /api/evaluation/clear
    POST /api/evaluation/verdict {call_id, score, correction}  (score + reground)
    GET  /api/models             ;  POST /api/models/active {model}
    GET  /api/rag/instances      ;  POST /api/rag/active {name}  ;  POST /api/rag/delete {name}
    GET  /api/retrieval          ;  POST /api/retrieval {key,value}  ;  POST /api/retrieval/reset
    GET  /api/system-prompt      ;  POST /api/system-prompt {text}
    GET  /api/system-prompts     ;  POST /api/system-prompts/save {name,text}
                                 ;  POST /api/system-prompts/delete {name}
    GET  /healthz

Run with:  python -m web.server   (or  python web/main.py)
"""
from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib import parse

from .aggregator import (
    activity_detail,
    clear_evaluation_queue,
    clear_recent_activity,
    delete_rag_instance,
    delete_system_prompt_preset,
    enqueue,
    evaluation_queue,
    get_models,
    get_rag_instances,
    get_retrieval_settings,
    get_system_prompt,
    iso_now,
    list_system_prompts,
    recent_activity,
    reset_retrieval_settings,
    save_system_prompt_preset,
    set_active_model,
    set_active_rag,
    set_retrieval_setting,
    set_system_prompt,
    simulate_prompt,
    submit_verdict,
)
from .config import Config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/activity": ("activity.html", "text/html; charset=utf-8"),
    "/activity.html": ("activity.html", "text/html; charset=utf-8"),
    "/evaluation": ("evaluation.html", "text/html; charset=utf-8"),
    "/evaluation.html": ("evaluation.html", "text/html; charset=utf-8"),
    "/settings": ("settings.html", "text/html; charset=utf-8"),
    "/settings.html": ("settings.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/activity.js": ("activity.js", "application/javascript; charset=utf-8"),
    "/evaluation.js": ("evaluation.js", "application/javascript; charset=utf-8"),
    "/settings.js": ("settings.js", "application/javascript; charset=utf-8"),
    "/aiar-logo.png": ("aiar-logo.png", "image/png"),
    "/favicon.png": ("aiar-logo.png", "image/png"),
    "/apple-touch-icon.png": ("aiar-logo.png", "image/png"),
}


class WatcherHandler(BaseHTTPRequestHandler):
    config = Config.load()

    def do_GET(self) -> None:  # noqa: N802
        parsed = parse.urlparse(self.path)
        path = parsed.path
        query = parse.parse_qs(parsed.query)

        if path in _STATIC:
            name, mime = _STATIC[path]
            self._serve_file(self.config.static_dir / name, mime)
            return
        if path == "/LICENSE":
            self._serve_file(_PROJECT_ROOT / "LICENSE", "text/plain; charset=utf-8")
            return
        if path == "/NOTICE":
            self._serve_file(_PROJECT_ROOT / "NOTICE", "text/plain; charset=utf-8")
            return
        if path == "/api/activity":
            self._serve_json(HTTPStatus.OK, recent_activity(self.config))
            return
        if path == "/api/evaluation/queue":
            self._serve_json(HTTPStatus.OK, evaluation_queue(self.config))
            return
        if path == "/api/activity/detail":
            call_id = (query.get("call_id") or [""])[0].strip()
            if not call_id:
                self._serve_json(HTTPStatus.BAD_REQUEST, {"error": "missing_call_id"})
                return
            payload = activity_detail(self.config, call_id)
            status = HTTPStatus.OK if payload.get("found") else HTTPStatus.NOT_FOUND
            self._serve_json(status, payload)
            return
        if path == "/api/models":
            self._serve_json(HTTPStatus.OK, get_models())
            return
        if path == "/api/rag/instances":
            self._serve_json(HTTPStatus.OK, get_rag_instances())
            return
        if path == "/api/retrieval":
            self._serve_json(HTTPStatus.OK, get_retrieval_settings())
            return
        if path == "/api/system-prompt":
            self._serve_json(HTTPStatus.OK, get_system_prompt())
            return
        if path == "/api/system-prompts":
            self._serve_json(HTTPStatus.OK, list_system_prompts())
            return
        if path == "/healthz":
            from aiar.llm import active_model, healthcheck
            from aiar.rag import store
            rag = store.health()
            ollama_ok = healthcheck()
            payload = {
                "status": "ok" if ollama_ok and rag.get("store_ready") else "degraded",
                "generated_at": iso_now(),
                "service": "aiar-watcher",
                "ollama_reachable": ollama_ok,
                "active_model": active_model(),
                "rag": rag,
            }
            self._serve_json(HTTPStatus.OK, payload)
            return
        self._serve_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": self.path})

    def do_POST(self) -> None:  # noqa: N802
        parsed = parse.urlparse(self.path)
        body = self._read_json()
        if body is None:
            self._serve_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return

        if parsed.path == "/api/simulate":
            prompt = str((body or {}).get("prompt") or "").strip()
            if not prompt:
                self._serve_json(HTTPStatus.BAD_REQUEST, {"error": "missing_prompt"})
                return
            try:
                result = simulate_prompt(
                    prompt,
                    rag=bool(body.get("rag", True)),
                    think=bool(body.get("think", False)),
                    reground=bool(body.get("reground", False)),
                    judge=bool(body.get("judge", True)),
                    instance=(str(body["instance"]).strip()
                              if body.get("instance") else None),
                    model=(str(body["model"]).strip()
                           if body.get("model") else None),
                    system=(body["system"] if body.get("system") is not None else None),
                )
            except Exception as exc:  # ollama down, etc.
                self._serve_json(HTTPStatus.BAD_GATEWAY,
                                 {"error": "harness_failed", "detail": str(exc)})
                return
            self._serve_json(HTTPStatus.OK, result)
            return

        if parsed.path == "/api/activity/evaluate":
            call_id = str((body or {}).get("call_id") or "").strip()
            if not call_id:
                self._serve_json(HTTPStatus.BAD_REQUEST, {"error": "missing_call_id"})
                return
            result = enqueue(self.config, call_id)
            self._respond_result(result)
            return

        if parsed.path == "/api/evaluation/verdict":
            call_id = str((body or {}).get("call_id") or "").strip()
            score = (body or {}).get("score")
            correction = str((body or {}).get("correction") or "")
            if not call_id:
                self._serve_json(HTTPStatus.BAD_REQUEST, {"error": "missing_call_id"})
                return
            if not isinstance(score, int):
                self._serve_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_score"})
                return
            result = submit_verdict(self.config, call_id, score, correction)
            self._respond_result(result)
            return

        if parsed.path == "/api/evaluation/clear":
            result = clear_evaluation_queue(self.config)
            self._respond_result(result)
            return

        if parsed.path == "/api/activity/clear":
            self._respond_result(clear_recent_activity())
            return

        if parsed.path == "/api/models/active":
            self._respond_result(set_active_model(str((body or {}).get("model") or "")))
            return

        if parsed.path == "/api/rag/active":
            self._respond_result(set_active_rag(str((body or {}).get("name") or "")))
            return

        if parsed.path == "/api/rag/delete":
            self._respond_result(delete_rag_instance(str((body or {}).get("name") or "")))
            return

        if parsed.path == "/api/retrieval":
            self._respond_result(set_retrieval_setting(
                str((body or {}).get("key") or ""), (body or {}).get("value")))
            return

        if parsed.path == "/api/retrieval/reset":
            self._respond_result(reset_retrieval_settings())
            return

        if parsed.path == "/api/system-prompt":
            text = (body or {}).get("text")
            self._respond_result(set_system_prompt("" if text is None else str(text)))
            return

        if parsed.path == "/api/system-prompts/save":
            self._respond_result(save_system_prompt_preset(
                str((body or {}).get("name") or ""),
                str((body or {}).get("text") or "")))
            return

        if parsed.path == "/api/system-prompts/delete":
            self._respond_result(delete_system_prompt_preset(
                str((body or {}).get("name") or "")))
            return

        self._serve_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": self.path})

    # ---- helpers ----------------------------------------------------------

    def _read_json(self) -> Optional[dict]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return None

    def _respond_result(self, result: dict) -> None:
        status = HTTPStatus(int(result.get("status") or (200 if result.get("ok") else 502)))
        payload = result.get("data") if result.get("ok") else {
            "error": result.get("error") or "failed", "detail": result.get("data")}
        self._serve_json(status, payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _serve_file(self, path: Path, content_type: Optional[str] = None) -> None:
        if not path.exists():
            self._serve_json(HTTPStatus.NOT_FOUND,
                             {"error": "asset_not_found", "path": str(path)})
            return
        body = path.read_bytes()
        mime = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run() -> None:
    config = Config.load()
    WatcherHandler.config = config
    # Warm the RAG store once so simulated prompts retrieve immediately.
    try:
        from aiar.rag import store
        store.init()
    except Exception:
        pass
    server = ThreadingHTTPServer((config.host, config.port), WatcherHandler)
    print(f"AIAR watcher serving on http://{config.host}:{config.port}")
    server.serve_forever()


if __name__ == "__main__":  # pragma: no cover
    run()
