"""Installation + environment checks for AIAR.

The goal is a single command a fresh user or AI agent can run after install to
confirm whether AIAR is actually ready for a demo:

    aiar-doctor
    aiar-doctor --json

This checks Python compatibility, core/RAG optional dependencies, Ollama
reachability, installed model visibility, and whether the local example corpus
can be ingested. It is intentionally read-only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List


_RAG_MODULES = ("chromadb", "sentence_transformers", "rank_bm25")
_CORE_MODULES = ("requests",)
_PASS = "pass"
_WARN = "warn"
_FAIL = "fail"


def _status(name: str, level: str, message: str, **extra: Any) -> Dict[str, Any]:
    item: Dict[str, Any] = {"name": name, "level": level, "message": message}
    item.update(extra)
    return item


def _module_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_checks() -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    py_ok = sys.version_info >= (3, 10)
    checks.append(_status(
        "python",
        _PASS if py_ok else _FAIL,
        f"Python {platform.python_version()} detected"
        + ("" if py_ok else " (AIAR requires Python 3.10+)"),
        version=platform.python_version(),
    ))

    missing_core = [m for m in _CORE_MODULES if not _module_present(m)]
    checks.append(_status(
        "core_deps",
        _PASS if not missing_core else _FAIL,
        "Core dependencies installed" if not missing_core
        else f"Missing core dependencies: {', '.join(missing_core)}",
        missing=missing_core,
    ))

    missing_rag = [m for m in _RAG_MODULES if not _module_present(m)]
    checks.append(_status(
        "rag_deps",
        _PASS if not missing_rag else _FAIL,
        "RAG dependencies installed" if not missing_rag
        else ("Missing RAG dependencies: " + ", ".join(missing_rag)
              + " (install with `pip install -e '.[rag]'`)"),
        missing=missing_rag,
    ))

    from aiar.llm import active_model, healthcheck, list_models

    ollama_ok = healthcheck()
    checks.append(_status(
        "ollama",
        _PASS if ollama_ok else _FAIL,
        "Ollama is reachable" if ollama_ok
        else "Ollama is not reachable at OLLAMA_URL",
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate"),
    ))

    active = active_model()
    installed = list_models(show_all=True) if ollama_ok else []
    installed_names = sorted({str(m.get("name") or "") for m in installed if m.get("name")})
    model_ok = bool(active and active in installed_names) if ollama_ok else False
    checks.append(_status(
        "model",
        _PASS if model_ok else (_WARN if not ollama_ok else _FAIL),
        (f"Active model {active!r} is installed" if model_ok else
         (f"Cannot verify active model {active!r} because Ollama is unreachable"
          if not ollama_ok else
          f"Active model {active!r} is not installed in Ollama")),
        active_model=active,
        installed_models=installed_names[:20],
    ))

    if not missing_rag:
        from aiar.rag import store
        store.init()
        rag_ready = store.is_ready()
        checks.append(_status(
            "rag_store",
            _PASS if rag_ready else _FAIL,
            "RAG store initialized successfully" if rag_ready
            else "RAG store failed to initialize",
        ))
    else:
        checks.append(_status(
            "rag_store",
            _WARN,
            "Skipped RAG store initialization because RAG dependencies are missing",
        ))

    examples = Path(__file__).resolve().parents[1] / "examples" / "docs"
    if examples.is_dir():
        from aiar.rag.ingest import ingest_folder
        try:
            chunks = ingest_folder(examples)
            checks.append(_status(
                "example_corpus",
                _PASS if chunks else _WARN,
                (f"Example corpus is present and ingestable ({len(chunks)} chunk(s))"
                 if chunks else
                 "Example corpus directory exists but ingest produced no chunks"),
                path=str(examples),
                chunks=len(chunks),
            ))
        except Exception as exc:  # pragma: no cover - defensive
            checks.append(_status(
                "example_corpus",
                _WARN,
                f"Example corpus ingest check failed: {exc}",
                path=str(examples),
            ))
    else:
        checks.append(_status(
            "example_corpus",
            _WARN,
            "Example corpus directory is missing",
            path=str(examples),
        ))

    overall = _PASS
    if any(c["level"] == _FAIL for c in checks):
        overall = _FAIL
    elif any(c["level"] == _WARN for c in checks):
        overall = _WARN

    return {
        "overall": overall,
        "checks": checks,
    }


def _render_text(report: Dict[str, Any]) -> str:
    badge = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    lines = [f"AIAR doctor: {badge.get(report['overall'], report['overall']).upper()}"]
    for item in report.get("checks") or []:
        lines.append(f"- {badge.get(item['level'], item['level']).upper():4s} {item['name']}: {item['message']}")
    if report["overall"] == _FAIL:
        lines.append("")
        lines.append("Suggested next step: follow README.md / PLAYBOOK.md and rerun `aiar-doctor`.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="aiar-doctor",
        description="Verify whether this AIAR install is ready for a local demo.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
    args = parser.parse_args(argv)

    report = run_checks()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render_text(report))
    return 0 if report["overall"] != _FAIL else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
