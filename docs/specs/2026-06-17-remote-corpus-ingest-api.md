# AIAR Spec — Remote Corpus Ingest & Instance API (implementation)

**Status:** Draft
**Repo:** this one (`github.com/wiggins-j/aiar`), targets the `aiar` package.
**Driver:** Errorta F088 "AIAR owns the corpus." Errorta runs on the Mac; the
AIAR instance lives on `watchdog`. Errorta must create/own/query a corpus in the
remote AIAR over the network.
**Consumer spec it answers:** `Errorta/docs/specs/AIAR-remote-corpus-ingest-api.md`
(authored by the Errorta side). That document defines the *contract* Errorta
wants; **this** document is the AIAR-side implementation design grounded in the
actual code (`service.py`, `store.py`, `ingest.py`, `instances.py`).

This spec specifies the change. It does not implement it.

---

## 1. Summary

Add an **authenticated remote ingest + instance-management API** to the existing
FastAPI service (`aiar/harness/service.py`). Today that service exposes
**query/eval only** and has **no auth**. Every capability the consumer wants
already exists as in-process Python in `store.py` / `ingest.py`; the work is
"wrap these functions in routes + add a bearer-token guard + add a small ingest
job model," plus closing a few **soundness gaps** that only matter once writes
arrive over the network (§5).

Scope is intentionally close to the consumer spec's §4 + §6. AIAR owns
*chunk + embed + store + retrieve*; Errorta keeps collection/extraction.

---

## 2. What already exists (verified against code)

The consumer spec's "current surface" section is accurate. Confirmed in this repo:

- **`aiar/rag/store.py`**
  - `create_instance(name, *, display_name=None, query_rewrite=None, rerank_model=None) -> str` — idempotent, returns slug, opens the (empty) collection. `store.py:243`
  - `publish_instance(name) -> None` `store.py:259`
  - `delete_instance(name) -> dict` — refuses `default`/`none`, purges collection + BM25 + grounding. `store.py:267`
  - `list_instances() -> List[dict]` → `[{name, display_name, status, chunk_count, active}]`. `store.py:347`
  - `health(*, instance=None) -> dict` — cheap, **deliberately does NOT load the embedder** (so `embedder_ready` reflects "loaded," not "loadable"). `store.py:366`
  - `chunk_count(*, instance=None) -> Optional[int]` `store.py:526`
  - `add(chunks, *, instance) -> int` — embeds (MiniLM) + stores, **dedups per `source` via `document_hash`**, batches under ChromaDB's ~5461 cap, returns **count added only**. `store.py:406`
- **`aiar/rag/ingest.py`**
  - `Chunk(source, title, chunk_index, text, category="general", metadata={}, page_span=None)` `ingest.py:43`
  - `_chunk_text(source, title, text, category, metadata=None, page_nums=None) -> List[Chunk]` — paragraph-boundary chunker (target 1600 chars, 200 overlap). **`page_nums` must be 1:1 with paragraphs** (`text.split("\n\n")`), not a per-document `[start,end]`. `ingest.py:110`
  - `ingest_file(path, *, category) -> List[Chunk]` — sets `metadata["document_hash"] = sha256(text)`; PDF path builds per-paragraph page numbers. `ingest.py:169`
- **`aiar/harness/service.py`** — FastAPI `app`, routes: `POST /eval/prompt`, `POST /services/prompt`, `POST /reground`, `GET /healthz`, `GET /services/meta`. **No auth.** `store.init()` on startup. Run via `uvicorn aiar.harness.service:app` (loopback `127.0.0.1:8766` on `watchdog`).
- **`aiar/rag/instances.py`** — `slugify()`, registry with `status ∈ {draft, published}`, `default` is born-aware and undeletable; `rag_<slug>` collection naming.
- **Query selection already works per-instance**: `answer_prompt(..., instance=...)` (`pipeline.py:103`) and `ServicePromptRequest.instance` (`service.py:54`). A freshly-published instance is selectable with no pipeline change.

**Deploy note (relevant now):** the recent cleanup deleted only the `~/aiar-dev`
*code/venv* directory on `watchdog`. The **corpus data survives** at
`~/.aiar-dev/knowledge` (the live `db_path`; holds `aerospace` ≈ 21,394 chunks).
A redeploy that points `AIAR_DB_PATH` back at `~/.aiar-dev/knowledge` regains all
instances untouched. The new install should be a proper git checkout (not the
prior untracked `0.1.0` dir); this repo is `0.2.0`.

---

## 3. Design

### 3.1 Where the code lives

New module **`aiar/harness/admin_routes.py`** exporting an `APIRouter`, mounted in
`service.py` via `app.include_router(...)`. Keeps the thin query service readable
and isolates the auth dependency. Job state lives in a new
**`aiar/harness/ingest_jobs.py`** (in-process registry, see §3.4).

No new retrieval logic. No change to existing routes' behaviour (AC#6).

### 3.2 Auth (bearer token) — §6 of the consumer spec

- New dependency `require_token` (FastAPI `Depends`). Reads `AIAR_SERVICE_TOKEN`
  from env/config at request time.
- Applied to **all mutating routes** (`/instances*` writes, all ingest). Read-only
  `/healthz`, `/services/meta`, `/services/prompt`, `/eval/prompt` are unchanged
  and stay open (loopback + tunnel is the trust boundary for reads).
- Behaviour:
  - No/empty/wrong `Authorization: Bearer <token>` on a mutating route → **401**.
  - **If `AIAR_SERVICE_TOKEN` is unset, mutating routes are disabled → 503** with
    a clear message ("remote ingest disabled: set AIAR_SERVICE_TOKEN"). This is
    fail-closed: an unconfigured box never silently accepts unauthenticated
    writes. (Reads keep working, so existing query deployments are unaffected.)
  - Token compared with `hmac.compare_digest` (constant-time).
- Transport unchanged: service stays bound to `127.0.0.1`; Errorta reaches it via
  `ssh -L 8766:127.0.0.1:8766 senditai`. We do **not** add `0.0.0.0` binding.

### 3.3 Readiness gating (closes a real soundness gap — see §5)

A `_require_store_ready()` dependency on every ingest/instance-write route:
- `store.is_ready()` is False → **503** `{code:"store_unavailable"}`.
- Force the embedder to load and 503 if it can't. `store.health()` will **not**
  do this (by design). We need a new tiny store helper:
  **`store.ensure_writable() -> None`** that calls `_ensure_embedder()` and raises
  a typed `StoreNotReady` if it returns False. The route maps that to 503.
  Without this, `store.add` silently returns `0` when the embedder fails to load
  (`store.py:411`) — indistinguishable from "all duplicates" — which would let a
  client believe a corpus built when nothing was embedded (violates AC#4).

### 3.4 Ingest job model

`aiar/harness/ingest_jobs.py`: an in-process dict `{job_id: JobRecord}` guarded by
a lock; `JobRecord` mirrors the consumer's bootstrap-job shape:

```
job_id, instance, status: "running"|"done"|"failed",
documents_total, chunks_added, duplicates, errors: [...],
started_at, ended_at
```

- `job_id` = `uuid4` hex (note: AIAR scripts can't use `random`/`uuid` freely in
  some sandboxes, but the *service* process is a normal runtime — `uuid4` is fine
  here; this is not a Workflow script).
- **v1 execution = synchronous, async-shaped contract.** The POST runs the ingest
  inline (FastAPI `BackgroundTasks` is acceptable but adds failure-visibility
  cost), records the terminal job, and returns `202 {job_id, accepted, instance}`.
  The job is already `done`/`failed` by the time the client polls. This keeps the
  contract async (so we can move to true background execution later without a
  client change) while avoiding a job runner in v1. **Recommended** — see Open
  Question 3. For brief-sized batches (thousands of docs) the client pages by
  staying under the per-request caps (§3.7); each page is its own job.
- Jobs are in-memory only in v1 (lost on restart). The consumer spec asks for
  "recoverable if the process dies" — deferred to phase 2 (persist to
  `~/.aiar*/ingest-jobs.jsonl`); for v1, a lost job is re-derivable because ingest
  is idempotent (re-POST the same `doc_id` batch → 0 new chunks).

### 3.5 Document ingest — chunk + embed server-side (§4.2)

`POST /instances/{instance}/documents`, body per the consumer spec. Per document:

1. Resolve `instance` (create-on-first-write is **not** done here; the instance
   must already exist via `POST /instances` → 404 if unknown, so a typo never
   spawns a stray corpus — see Open Question 4).
2. **Idempotency by `doc_id`:** keep a per-instance manifest
   `{doc_id: document_hash}` (persisted beside the registry, e.g.
   `<base>/knowledge/ingest_manifest.json`). If `doc_id` is present with the same
   `sha256(text)`, **skip re-chunking** and count the doc as a duplicate. This is
   the optimisation the consumer spec describes; the store's per-`source` +
   `document_hash` dedup (`store.py:434`) remains the correctness backstop.
3. Chunk with `_chunk_text(source=doc.source, title=doc.title, text=doc.text,
   category=doc.category, metadata={**doc.metadata, "document_hash": sha256(text),
   "doc_id": doc_id}, page_nums=<see §3.6>)`.
4. `store.add(chunks, instance=instance)` → returns added count.
5. Accumulate `chunks_added`; `duplicates = sum(candidate_chunks) - chunks_added`
   (derivable since `add` returns only `added`). A per-doc failure appends to
   `errors[]` and never aborts the batch (§7 of consumer spec).
6. If `publish: true`, `store.publish_instance(instance)` after the batch.

`Chunk.source` MUST be the document's `source` field (stable provenance) so the
store's per-`source` dedup grouping works; `doc_id` rides in metadata.

### 3.6 Page spans — resolve the contract mismatch (Open Question 5)

The consumer schema sends document-level `page_spans: [[start,end], ...]`, but
`_chunk_text` consumes **`page_nums` aligned 1:1 with paragraphs**, and emits each
chunk's `page_span` as `(min,max)` of its paragraphs' pages. A list of
`[start,end]` ranges cannot be mapped to per-paragraph pages without knowing where
each page's text begins. Options:

- **(A) Recommended:** extend the document schema with an optional
  `pages: [{page: int, text: str}, ...]` (mirrors `ingest._pdf_to_text` output).
  When present, the server concatenates page texts into the document body and
  derives `page_nums` per paragraph exactly like `ingest_file`'s PDF path
  (`ingest.py:196-213`) — faithful round-trip, reuses existing logic.
- **(B) Fallback:** if only flat `text` is sent (no `pages`), ingest with
  `page_nums=None` → `page_span=None`. F013 source-jump degrades to whole-doc, no
  crash.
- The `/chunks` passthrough (§4.4 / phase 2) is the only path where a client sets
  `page_span` per chunk directly.

Decision needed from Errorta: can the Mac-side extractor emit `pages[]`? If yes,
F013 keeps working remotely with zero retrieval changes.

### 3.7 Limits & errors (§7 of consumer spec)

- Per-request caps: **≤ 200 documents** and **≤ 8 MB** body (FastAPI/uvicorn limit
  + an explicit guard) → **413** over cap; client pages.
- `store.add` already batches embeddings under ChromaDB's ~5461 cap
  (`store.py:449`); no API-side chunk batching needed.
- Never embed-on-read; never accept client vectors (no field for them exists, and
  none will be added).

---

## 4. HTTP API (final shape)

All mutating routes: `Depends(require_token)` + `Depends(_require_store_ready)`.
Slugs via `instances.slugify`.

### 4.1 Instance management
```
POST   /instances                    {name, display_name?, query_rewrite?, rerank_model?}
        -> 200 {instance, status:"draft", created: true|false}       # create_instance (idempotent)
GET    /instances                    -> 200 {instances:[{name, display_name, status, published, chunk_count, active}]}
GET    /instances/{instance}/health  -> 200 {store_ready, embedder_ready, embedding_model, chunk_count, published, ...}
POST   /instances/{instance}/publish -> 200 {instance, published: true}
DELETE /instances/{instance}         -> 200 {deleted: true, active: "<resolved>"}   # 400 for default/none
```
`GET /instances` reads `list_instances()` and adds `published = (status ==
"published")`. `created` is computed by checking `list_instances()` membership
before the idempotent `create_instance`.

### 4.2 Document ingest (primary)
```
POST /instances/{instance}/documents
Body: { "documents": [ {doc_id, source, title, category?, text, metadata?, pages?, page_spans?} ], "publish": false }
  -> 202 {job_id, accepted: N, instance}     # 404 unknown instance, 413 over caps, 401/503 per guards
```

### 4.3 Job status
```
GET /instances/{instance}/ingest-jobs/{job_id}
  -> 200 {job_id, status, documents_total, chunks_added, duplicates, errors:[...], started_at, ended_at}
  -> 404 unknown job_id
```

### 4.4 Phase 2 (optional, not v1)
```
POST /instances/{instance}/files   (multipart)  -> 202 {job_id}   # server runs ingest_file()
POST /instances/{instance}/chunks  {chunks:[...], publish?} -> 202 {job_id}   # passthrough to store.add
```

### 4.5 Query — unchanged
`POST /services/prompt` with `instance` already serves retrieval against a
published instance. No change beyond confirming a freshly-created instance is
selectable (it is — `_handle` auto-opens collections, `store.py:210`).

---

## 5. Soundness gaps to close (the part that's NOT just wrapping)

1. **Silent embedder failure → false success.** `store.add` returns `0` both when
   "all duplicates" and when the embedder failed to load. Add
   `store.ensure_writable()` (§3.3) and gate every ingest route on it → 503, never
   a 0-chunk "success." (AC#4.)
2. **`duplicates` not surfaced.** `add` returns only `added`. Either derive
   `duplicates = candidates - added` in the route (sufficient for v1), or
   optionally widen `add` to return `{"added": n, "skipped": m}` (cleaner, but a
   signature change with in-process callers — defer).
3. **Create-on-write footgun.** `_handle`/`create_instance` auto-create instances
   on first reference. The ingest route must require the instance to pre-exist
   (404 otherwise) so a typo'd path can't silently spawn a corpus. (Open Q4.)
4. **`embedder_ready` semantics.** `health()` reports whether the embedder is
   *loaded*, not loadable. The per-instance health route should call
   `ensure_writable()` first (or document the distinction) so a remote client's
   readiness check is meaningful before a large push.

---

## 6. Testing

**Where things run (standing constraint):** the MacBook is **development-only** —
code, specs, and unit tests with a **mocked embedder/store** run there. Anything
that loads the real MiniLM embedder, writes real ChromaDB, runs the live service,
or ingests a real corpus runs on **`watchdog`** (`ssh senditai`), never locally.
So the suite below splits in two:

- **Mac (dev, mocked):** all unit tests — auth, instance CRUD, job bookkeeping,
  idempotency logic, caps/errors — with the embedder monkeypatched. Fast, no
  model load. This is the default `pytest tests/ -k admin` run during dev.
- **`watchdog` (integration, real):** the create → ingest → publish → query
  round-trip against the real embedder + ChromaDB, and the SSH-tunnel smoke test
  from the Mac. Run on the server after deploy, not locally.

Match `tests/test_service.py` conventions: `shared_runtime_state` fixture
(`AIAR_BASE_DIR` + `AIAR_DB_PATH` in `tmp_path`, `store.reset_for_testing(base=)`),
FastAPI `TestClient`, monkeypatch `store`/embedder where embedding is slow.

- `test_admin_auth.py`: mutating route → 401 without token; 200 with; 503 when
  `AIAR_SERVICE_TOKEN` unset.
- `test_admin_instances.py`: create (idempotent `created` flag) → list → publish →
  health → delete; `default`/`none` delete → 400.
- `test_admin_ingest.py`: ingest a small batch → job `done` with correct
  `chunks_added`; re-post same `doc_id` → `chunks_added == 0`, `duplicates > 0`
  (AC#3); unknown instance → 404; over-cap → 413; bad doc → recorded in `errors[]`,
  rest succeed; store-not-ready → 503 (no false success, AC#4).
- `test_admin_query_roundtrip.py`: create → ingest → publish → `POST
  /services/prompt` returns an answer grounded in the ingested text; `aerospace`
  (a second instance) untouched (AC#1, #2, #6).
- Page spans: if §3.6(A) adopted, assert `pages[]` round-trips to chunk
  `page_span` (extends `tests/test_chunk_page_span.py`).

Run: `pytest tests/ -k admin` (existing suite: `pytest tests/`).

---

## 7. Acceptance criteria (maps to consumer §9)

1. Remote client w/ token: create → ingest batch → poll `done` → publish →
   `/services/prompt` returns grounded answers. → §4 + integration test.
2. New instance appears in `GET /instances` and per-instance health with correct
   `chunk_count`, `aerospace` untouched. → §4.1.
3. Re-posting same `doc_id` adds 0 chunks. → §3.5 idempotency + manifest.
4. Mutating routes: 401 w/o token, 503 when store/embedder not ready, **never**
   success with 0 chunks stored. → §3.2, §3.3, §5.1.
5. All embedding server-side MiniLM; no route accepts client vectors. → §3.5; no
   vector field exists.
6. Existing query/eval routes + `aerospace` unchanged. → new router only; no edits
   to existing handlers; regression test.
7. Loopback bind preserved; works over the SSH tunnel. → §3.2; no bind change.
8. `GET /healthz` advertises `"remote_ingest": true` once the ingest routes are
   mounted (a query-only AIAR reports `false`). The Errorta client gates
   ingest-availability on this marker, not on `store_ready`/`embedder_ready`, so
   a healthy-but-query-only host doesn't look ingest-capable and 404 later. →
   `service._remote_ingest_mounted()`, derived from the mounted routes.

---

## 8. Rollout

**Topology:** dev on the MacBook, everything else on `watchdog`. The Mac never
builds, embeds, runs the service, or holds a corpus — it edits code and runs
mocked unit tests. The build/run/deploy target is always the server.

1. **(Mac, dev)** Land routes + auth + jobs + `store.ensure_writable()` + mocked
   unit tests in this repo; push to GitHub.
2. **(`watchdog`, deploy)** Deploy as a **proper git checkout** (not an untracked
   dir like the old `0.1.0` install) with `AIAR_DB_PATH=~/.aiar-dev/knowledge`
   (regains `aerospace`) and `AIAR_SERVICE_TOKEN` set. Add a **systemd unit + env
   file** so the service survives reboot — the old install had none and ran as a
   bare process. Run the real-embedder integration tests here.
3. **(`watchdog`)** Service stays bound to `127.0.0.1:8766`; Errorta on the Mac
   reaches it over `ssh -L 8766:127.0.0.1:8766 senditai`.
4. Errorta builds `RemoteAiarCorpusAdapter` (consumer §8) against this contract.

---

## 9. Open questions for the operator (decisions baked as recommendations above)

1. **Token model:** single static `AIAR_SERVICE_TOKEN` for v1 (recommended), or
   per-client tokens (Errorta F009 pairing) later? Spec assumes static + fail-closed.
2. **Server-side briefs:** confirmed out of scope — AIAR is the ingest sink, not
   the collector? (Spec assumes yes.)
3. **Sync vs async:** v1 ships **synchronous execution behind an async-shaped job
   contract** (§3.4). OK, or do you want true background execution + job
   persistence in v1?
4. **Reserved names / create-on-write:** spec **requires instances to pre-exist**
   for ingest (404 otherwise) and suggests protecting curated names
   (`default`, `aerospace`). Any quota/namespacing for Errorta-created instances
   (e.g. an `errorta-` prefix)?
5. **Page spans:** needs Errorta to emit `pages: [{page, text}]` for faithful
   F013 round-trip (§3.6 option A); otherwise `page_span=None`. Can the Mac
   extractor provide per-page text?
