# Implementation Plan — Remote Corpus Ingest & Instance API

**Spec:** [2026-06-17-remote-corpus-ingest-api.md](2026-06-17-remote-corpus-ingest-api.md)
**Consumer:** `Errorta/docs/specs/AIAR-remote-corpus-ingest-api.md` (client = PR #37, done)
**Decisions:** all 5 locked (table below). Build target: `watchdog`. Dev: Mac (mocked).

| # | Decision (locked) |
|---|---|
| 1 | Static `AIAR_SERVICE_TOKEN`; mutating routes **503** if unset, **401** on bad token |
| 2 | AIAR = **ingest sink only** (Errorta collects/extracts) |
| 3 | **Sync behind async contract** — POST runs inline, returns `202 {job_id}`, already `done`/`failed` on poll; background runner deferred |
| 4 | Instance must **pre-exist (404)**; `default`/`aerospace` **delete-protected**; `errorta-` prefix optional (not enforced v1) |
| 5 | `pages:[{page,text}]` → AIAR derives `page_span`; flat `text` → `None` |

Work is split into 8 steps, each a self-contained commit. Steps 1–7 are Mac dev +
mocked unit tests; step 8 is `watchdog` integration + deploy. No edits to existing
query/eval handlers (AC#6) — all new code is additive.

---

## Step 1 — `store` write-readiness + reserved-name protection

**File:** `aiar/rag/store.py`

1. Add a typed error and a public guard:
   ```python
   class StoreNotReady(RuntimeError):
       """Raised when an ingest/write is attempted but the store or embedder
       cannot serve it. Carries a machine code for the HTTP layer."""
       def __init__(self, code: str, message: str):
           super().__init__(message); self.code = code

   def ensure_writable() -> None:
       """Raise StoreNotReady if a write cannot succeed RIGHT NOW.
       Unlike health(), this *forces* the embedder to load so a remote client
       never gets a false 'success' from add() returning 0 on embedder failure."""
       if _registry is None and not _available:
           init()
       if not (is_ready() and _client is not None and _registry is not None):
           raise StoreNotReady("store_unavailable", "RAG store not initialised")
       if not _ensure_embedder():
           raise StoreNotReady("embedder_unavailable", "embedder failed to load")
   ```
   This is the fix for the silent-`0` gap (`add` returns 0 for both "all dups"
   and "embedder failed" — `store.py:411`). The route calls `ensure_writable()`
   before `add`, so embedder failure becomes 503, not a 0-chunk 200.

2. Reserved-name protection. Add a module constant and enforce in `delete_instance`:
   ```python
   RESERVED_INSTANCES = {"aerospace"}  # curated; never client-deletable
   ```
   In `delete_instance` (`store.py:267`), after canonicalising, raise
   `ValueError` if `name in RESERVED_INSTANCES` (mirrors the existing `default`
   guard). `none`/`default` already protected.

**Tests** (`tests/test_store_writable.py`, mocked): `ensure_writable()` raises
`StoreNotReady("embedder_unavailable")` when `_ensure_embedder` is monkeypatched
to False; passes when ready. `delete_instance("aerospace")` → `ValueError`.

**AC:** #4 (no false success), #2/#6 (aerospace protected).

---

## Step 2 — in-memory document ingest in `ingest.py`

**File:** `aiar/rag/ingest.py`

Add a function that chunks an **in-memory** document (text or pages) exactly like
`ingest_file`, so remote ingest reuses the same chunker + `document_hash` +
`page_span` logic — no divergence from the file path.

```python
def ingest_document(
    *, source: str, title: str,
    text: Optional[str] = None,
    pages: Optional[List[Dict[str, object]]] = None,   # [{page:int, text:str}, ...]
    category: str = "general",
    metadata: Optional[Dict[str, object]] = None,
) -> List[Chunk]:
    """Chunk an in-memory document into Chunks (server-side path for the HTTP API).

    - pages given  -> reconstruct body + per-paragraph page_nums (faithful F013
      page_span round-trip, same logic as the PDF branch of ingest_file).
    - text only    -> page_span=None.
    Sets metadata['document_hash']=sha256(body) so store.add dedup works.
    """
```

Implementation: refactor the PDF para-building block of `ingest_file`
(`ingest.py:196-213`) into a shared `_pages_to_body_and_nums(pages)` helper and
call it from both `ingest_file` and `ingest_document`. `document_hash` is computed
on the final body, identical to `ingest_file:219`. Decision #5: `pages` present →
derive `page_nums`; else `page_nums=None`.

**Tests** (`tests/test_ingest_document.py`): `pages=[{page:1,text:...},{page:2,...}]`
→ chunks carry correct `(min,max)` `page_span`; flat `text` → `page_span is None`;
`document_hash` stable for identical input. Extends `tests/test_chunk_page_span.py`
patterns.

**AC:** #5 (page_span round-trip), #5-invariant (server chunks with `_chunk_text`).

---

## Step 3 — bearer-token auth dependency

**File:** `aiar/harness/auth.py` (new)

```python
import hmac, os
from fastapi import Header, HTTPException

def _token() -> str | None:
    return os.environ.get("AIAR_SERVICE_TOKEN") or None

def require_token(authorization: str | None = Header(default=None)) -> None:
    cfg = _token()
    if not cfg:                       # decision #1: fail closed when unconfigured
        raise HTTPException(503, {"code": "ingest_disabled",
            "error": "remote ingest disabled: set AIAR_SERVICE_TOKEN"})
    sent = ""
    if authorization and authorization.lower().startswith("bearer "):
        sent = authorization[7:].strip()
    if not sent or not hmac.compare_digest(sent, cfg):
        raise HTTPException(401, {"code": "unauthorized", "error": "invalid token"})
```

Read at request time (not import) so the env can be set by the systemd unit and
flipped without code change. Constant-time compare.

**Tests** (`tests/test_admin_auth.py`): unset env → 503; wrong token → 401; correct
→ passes (dependency returns None).

**AC:** #4 (401/503 semantics), spec §3.2.

---

## Step 4 — ingest-job registry + doc_id manifest

**File:** `aiar/harness/ingest_jobs.py` (new)

- `@dataclass JobRecord`: `job_id, instance, status, documents_total,
  chunks_added, duplicates, errors: list, started_at, ended_at` + `to_dict()`.
- Module-level `_JOBS: Dict[str, JobRecord]` under a `threading.Lock`.
- `new_job(instance) -> JobRecord` (`uuid4().hex`, status `running`,
  `started_at=_iso_now()`); `finish(job, status)`; `get(instance, job_id)` →
  validates the job belongs to that instance (404 in route otherwise).
- Decision #3: jobs are in-memory only; lost on restart is acceptable because
  ingest is idempotent. (Phase-2 hook: optional append to
  `~/.aiar*/ingest-jobs.jsonl`.)

**Idempotency — no separate doc_id manifest (revised during implementation).**
The original plan added a `{instance: {doc_id: document_hash}}` manifest to skip
re-chunking. On closer reading of `store.add` (`store.py:434`), it **already**
skips re-embedding a document whose `document_hash` + chunk count are unchanged,
so the manifest only duplicated that — and worse, nothing cleared it on
`delete_instance`, so a delete→recreate→re-ingest would wrongly skip as "seen"
and silently build an empty corpus. Dropped it: the chunk-layer `document_hash`
dedup is the single source of truth (the spec's stated backstop). `doc_id` is
still carried into chunk metadata for provenance. `duplicates = candidate_chunks
- added` is derived per document.

**Tests** (`tests/test_ingest_jobs.py`): job lifecycle running→done; `get` with
wrong instance → None; manifest `seen`/`record` round-trip in `tmp_path`.

**AC:** #3 (job model), spec §3.4.

---

## Step 5 — admin router (models + routes)

**File:** `aiar/harness/admin_routes.py` (new) — `APIRouter`, all mutating routes
`dependencies=[Depends(require_token)]`.

Pydantic models: `CreateInstance{name, display_name?, query_rewrite?, rerank_model?}`,
`DocumentIn{doc_id, source, title, category="general", text?, pages?, metadata?}`,
`IngestRequest{documents: List[DocumentIn], publish: bool=False}`.

Routes (status codes explicit):

| Route | Wraps | Notes |
|---|---|---|
| `POST /instances` | `store.create_instance` | 200 `{instance, status:"draft", created}`. `created` = not in `list_instances()` pre-call |
| `GET /instances` | `store.list_instances` | adds `published = status=="published"` |
| `GET /instances/{id}/health` | `store.ensure_writable` then `store.health(instance=)` | forces embedder so `embedder_ready` is meaningful; 404 unknown id |
| `POST /instances/{id}/publish` | `store.publish_instance` | 404 unknown id |
| `DELETE /instances/{id}` | `store.delete_instance` | 400 for `default`/`aerospace`/`none`; 404 unknown |
| `POST /instances/{id}/documents` | see below | 202 `{job_id, accepted, instance}` |
| `GET /instances/{id}/ingest-jobs/{job_id}` | `ingest_jobs.get` | 404 unknown job |

`POST /documents` handler (decision #3 sync-inline):
1. `store.ensure_writable()` → 503 on `StoreNotReady` (map `.code`).
2. Resolve instance; **404 if it doesn't already exist** (decision #4 — do *not*
   auto-create here; check `store.list_instances()` / registry membership).
3. Enforce caps: ≤200 docs, ≤8 MB body → **413**.
4. `job = ingest_jobs.new_job(instance)`. Per document, in a try/except that
   appends to `job.errors` and continues (partial failure never aborts):
   - `doc_hash = sha256(text or joined pages)`; if `manifest.seen(...)` →
     count duplicate, skip.
   - `chunks = ingest.ingest_document(source, title, text=, pages=, category=,
     metadata={**meta, "doc_id": doc_id})`.
   - `added = store.add(chunks, instance=instance)`;
     `job.chunks_added += added`; `job.duplicates += len(chunks) - added`;
     `manifest.record(...)`.
5. If `publish` → `store.publish_instance(instance)`.
6. `ingest_jobs.finish(job, "done" | "failed")`; return `202 {job_id, accepted, instance}`.

**Tests** (`tests/test_admin_instances.py`, `tests/test_admin_ingest.py`, mocked
embedder): CRUD + idempotent `created`; default/aerospace delete → 400; ingest a
batch → job `done`, correct `chunks_added`; re-post same `doc_id` →
`chunks_added==0, duplicates>0` (AC#3); unknown instance → 404; over-cap → 413;
one bad doc recorded in `errors[]`, rest succeed; `ensure_writable` raising →
503 (no false success, AC#4).

**AC:** #1, #3, #4.

---

## Step 6 — wire router into the service

**File:** `aiar/harness/service.py`

- `from aiar.harness.admin_routes import router as admin_router`
- `app.include_router(admin_router)` after the existing routes.
- Bump `FastAPI(version=...)` from the stale `"0.1.0"` to match `pyproject`
  (`0.2.0`) while here (cosmetic, existing bug).
- Add `"remote_ingest": _remote_ingest_mounted()` to the `/healthz` body (AC#8) —
  the only edit to an existing route, additive (new key, existing keys unchanged).
  Derived from the mounted routes so a query-only build reports `false`.
- **No other changes** to `eval_prompt`, `service_prompt`, `reground`,
  `services_meta`. Loopback bind unchanged (run command unchanged).

**Tests** (`tests/test_service.py`): add a regression asserting the existing
routes still return their current shapes with the router mounted (AC#6).

**AC:** #6, #7 (loopback preserved).

---

## Step 7 — full mocked suite green on Mac

Run `pytest tests/` (and `-k admin` for the new files) on the Mac with the
embedder monkeypatched. Everything in steps 1–6 must pass here before deploy.
No real model load, no real ChromaDB writes locally (dev-only Mac).

---

## Step 8 — integration + deploy on `watchdog`

(Server only — the Mac never runs the live service or embeds.)

1. Push branch → open PR in `github.com/wiggins-j/aiar`.
2. On `watchdog`: deploy a **git checkout** (not an untracked dir like the old
   `0.1.0`). Set `AIAR_DB_PATH=~/.aiar-dev/knowledge` (regains `aerospace`,
   ~21,394 chunks survived the earlier cleanup) and `AIAR_SERVICE_TOKEN=<secret>`.
3. Add a **systemd unit + env file** for `uvicorn aiar.harness.service:app
   --host 127.0.0.1 --port 8766` (old install had no unit; this makes it
   reboot-durable).
4. Real-embedder integration test (run on server): create instance → `POST
   /documents` with `pages[]` → poll job `done` → publish → `POST /services/prompt`
   returns grounded answers with correct `page_span`; confirm `aerospace`
   untouched in `GET /instances`.
5. Tunnel smoke test from Mac: `ssh -L 8766:127.0.0.1:8766 senditai`, then
   Errorta's PR #37 adapter against `127.0.0.1:8766` — full create→ingest→query.

**AC:** #1, #2, #5, #6, #7 end-to-end.

---

## File-change summary

| File | Change |
|---|---|
| `aiar/rag/store.py` | + `StoreNotReady`, `ensure_writable()`, `RESERVED_INSTANCES`; guard `delete_instance` |
| `aiar/rag/ingest.py` | + `ingest_document()`, `_pages_to_body_and_nums()` (refactor from `ingest_file`) |
| `aiar/harness/auth.py` | **new** — `require_token` |
| `aiar/harness/ingest_jobs.py` | **new** — job registry + doc_id manifest |
| `aiar/harness/admin_routes.py` | **new** — models + 7 routes |
| `aiar/harness/service.py` | mount router; version bump; **no handler edits** |
| `tests/test_store_writable.py`, `test_ingest_document.py`, `test_admin_auth.py`, `test_ingest_jobs.py`, `test_admin_instances.py`, `test_admin_ingest.py` | **new** (mocked) |
| `tests/test_service.py` | + regression for unchanged routes |
| deploy (`watchdog`) | systemd unit + env file; `AIAR_DB_PATH`, `AIAR_SERVICE_TOKEN` |

## Invariants enforced (don't regress)
- Server-side MiniLM only; **no client-vector field exists anywhere**.
- Idempotent by `doc_id` (manifest) + `document_hash` (store.add backstop).
- 401 no token / 503 store-or-embedder-not-ready / **never** 200 with 0 chunks stored.
- Existing query/eval routes + `aerospace` corpus untouched.
- Service stays `127.0.0.1`-bound; reached over SSH tunnel.
