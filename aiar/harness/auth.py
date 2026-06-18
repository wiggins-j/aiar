"""Bearer-token auth for AIAR's remote admin corpus surface.

The harness service is loopback-bound and reached over an SSH tunnel, but the
ingest/instance-management routes are a *write* surface, so they require a bearer
token as defense-in-depth (and as a hard gate if the operator later binds to a
LAN/Tailscale address).

Policy (decision #1 — static token, fail-closed):
  * Token source is the ``AIAR_SERVICE_TOKEN`` env var, read at request time so
    it can be set/rotated by the service unit without a code change.
  * If it is unset, every mutating route is **disabled** (503) — an unconfigured
    box never silently accepts unauthenticated writes.
  * A missing/malformed/wrong token is **401**.

Public read-only service routes (``/healthz``, ``/services/meta``, query/eval)
do not use this dependency and stay open on the loopback interface. Corpus reads
that expose document contents (for example ``/instances/{id}/retrieve``) are
token-gated alongside ingest and instance management.
"""
from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException

_BEARER_PREFIX = "bearer "


def _configured_token() -> Optional[str]:
    token = os.environ.get("AIAR_SERVICE_TOKEN")
    return token or None  # treat empty string as unset


def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency guarding mutating routes. Returns ``None`` on success;
    raises 503 (ingest disabled) or 401 (bad token)."""
    configured = _configured_token()
    if not configured:
        raise HTTPException(
            status_code=503,
            detail={"code": "ingest_disabled",
                    "error": "remote ingest disabled: set AIAR_SERVICE_TOKEN"},
        )
    sent = ""
    if authorization and authorization.lower().startswith(_BEARER_PREFIX):
        sent = authorization[len(_BEARER_PREFIX):].strip()
    if not sent or not hmac.compare_digest(sent, configured):
        raise HTTPException(
            status_code=401,
            detail={"code": "unauthorized", "error": "invalid or missing token"},
        )
