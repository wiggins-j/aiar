from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    static_dir: Path
    host: str
    port: int
    # Where queued items and submitted verdicts are persisted (JSONL).
    queue_file: Path
    verdicts_file: Path
    # Score at/below which a correction (preferred response) is required.
    reason_threshold: int

    @classmethod
    def load(cls) -> "Config":
        project_root = Path(__file__).resolve().parent
        base = Path(os.environ.get("AIAR_BASE_DIR", "~/.aiar")).expanduser()
        return cls(
            static_dir=project_root / "static",
            host=os.environ.get("AIAR_WEB_HOST", "127.0.0.1"),
            port=_env_int("AIAR_WEB_PORT", 8088),
            queue_file=Path(os.environ.get(
                "AIAR_QUEUE_FILE", str(base / "eval-queue.jsonl"))).expanduser(),
            verdicts_file=Path(os.environ.get(
                "AIAR_VERDICTS_FILE", str(base / "verdicts.jsonl"))).expanduser(),
            reason_threshold=_env_int("AIAR_REASON_THRESHOLD", 7),
        )
