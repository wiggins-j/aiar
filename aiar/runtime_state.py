"""Small shared runtime state for cross-process AIAR entrypoints.

The watcher GUI, CLI, and optional HTTP service can run in separate processes.
Settings such as the active model, active RAG instance, and active harness
system prompt therefore need a tiny shared persistence layer so they stay
consistent across entrypoints on the same machine.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

_BASE_OVERRIDE: Optional[Path] = None


def _base_dir() -> Path:
    if _BASE_OVERRIDE is not None:
        return _BASE_OVERRIDE
    return Path(os.environ.get("AIAR_BASE_DIR", "~/.aiar")).expanduser()


def _state_path() -> Path:
    default = _base_dir() / "runtime-state.json"
    return Path(
        os.environ.get("AIAR_RUNTIME_STATE_FILE", str(default))
    ).expanduser()


def _read_state() -> Dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def get(key: str) -> Any:
    return _read_state().get(key)


def set_value(key: str, value: Any) -> None:
    state = _read_state()
    if value is None:
        state.pop(key, None)
    else:
        state[key] = value
    _write_state(state)


def reset_for_testing(*, base: Optional[Path] = None) -> None:
    global _BASE_OVERRIDE
    _BASE_OVERRIDE = Path(base) if base is not None else None
    path = _state_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass
