# AIAR integration contracts

Stable, schema-versioned contracts for apps built on AIAR. Each shape has **one
serializer** shared by the in-process Python API and the HTTP route, so local and
remote callers get byte-identical payloads.

## Conventions

- **Schema versioning.** Every payload carries `schema_version`
  (`aiar.<contract>.v<N>`). Bumps are additive within a version (new optional
  keys); a removed/renamed/retyped key is a new version.
- **Error bodies.** Typed errors return `{"code": <stable-slug>, "error": <msg>}`.
  Switch on `code`, not the prose. HTTP status accompanies but does not replace it.
- **Redaction.** Trace/telemetry surfaces redact prompt text and corpus bytes
  unless an explicit local debug flag (`AIAR_TRACE_DEBUG`) is set. Token counts
  and `call_id` are not redacted.

## 1. Pure retrieval — `aiar.retrieve.v1`

Source refs **without a generation call**.

**Ranking semantics:** pure retrieve returns **raw vector similarity** — it does
not run the answerer's hybrid/rerank/rewrite pipeline (those add latency and one
stage can call a model). Consequence: the chunks/order from `retrieve` are not
guaranteed to match the chunks that grounded a given `answer_prompt` call. For
"the exact sources behind this answer," request them from `answer_prompt`
(§4 `sources`), which reports the answerer's actual retrieved set. Use `retrieve`
for clean source lookup.

- HTTP: `GET/POST /instances/{instance}/retrieve` (`q`, `k` default 8 clamped to
  `[1,50]`, optional `category`). Bearer-token auth.
- Python: `aiar.rag.retrieve_chunks(query, *, instance, k=8, category=None)`.

```jsonc
{
  "schema_version": "aiar.retrieve.v1",
  "instance": "...", "query": "...", "k": 8, "count": 1,
  "score_kind": "cosine_similarity",   // response-level (constant across hits)
  "score_order": "desc",
  "hits": [{
    "chunk_id": "...", "source": "...", "title": "...", "text": "...",
    "score": 0.83, "chunk_index": 4, "category": "general",
    "page_span": [12, 13], "metadata": { }
  }]
}
```

Guarantees: never invokes a generation model; unknown/unpublished instance →
`404 unknown_instance`; blank `q` → `400 empty_query`; no hits → `200 count:0`.

## 2. Grounding — `aiar.grounding.v1`

Instance-scoped correction memory. `answer` (what was wrong) and `correction`
(the fix) are distinct fields, so a consumer can never store the answer in the
correction slot.

- `aiar.grounding.record_grounding(*, signature, verdict, correction="",
  instance=None, answer=None, prompt=None, source_chunks=None)`
- `aiar.grounding.lookup_grounding(*, signature, instance=None)`

```jsonc
{
  "schema_version": "aiar.grounding.v1",
  "id": "...", "signature": "...", "normalized": "...", "instance": "...",
  "verdict": "bad", "reason": "...", "correction": "the fix",
  "answer": "the original wrong answer", "prompt": "...",
  "source_chunks": ["chunk_id"], "failure_tags": [], "confidence": "high",
  "created_at": "..."
}
```

Legacy records (written before these fields existed) read back with `answer=null`
and the original text left in `correction` — never rewritten.

> Semantic lookup over groundings is not part of this version (it needs a vector
> index over groundings, which the JSON-file store does not have). The capability
> manifest reports `semantic_grounding: false`.

## 3. Ingest result + readiness — `aiar.ingest.v1`

So a consumer stops marking a file "ready" when AIAR can't retrieve it.

- Python (synchronous): `aiar.rag.ingest_documents(documents, *, instance,
  publish=False) -> IngestResult`.
- HTTP (asynchronous): `POST /instances/{instance}/documents` → `202 {job_id,...}`,
  poll `GET /instances/{instance}/ingest-jobs/{job_id}`. **The polled job IS an
  IngestResult** (plus transport-only keys).

```jsonc
{
  "schema_version": "aiar.ingest.v1",
  "instance": "...", "status": "done",   // running | done | failed
  "accepted": 3, "chunks_added": 42, "duplicates": 5,
  "errors": [{"doc_id": "...", "error": "..."}], "published": false
}
```

**`publish` defaults to `false`** (fail-closed). Partial-success semantics:

| Situation | status |
|---|---|
| chunks added, no errors | `done` |
| all duplicates, no errors (re-ingest) | `done` (idempotent) |
| some added **and** some doc errors | `done` (partial — surface errors) |
| 0 added **and** errors present | `failed` (never "ready") |
| requested publish failed | `failed` |

`health(instance)` reports `published`, `chunk_count`, `last_ingest_at`,
`last_ingest_error`.

## 4. Telemetry, trace & answer sources — `aiar.answer.v1`

`answer_prompt()` returns stable keys: `schema_version, call_id, instance, model,
system_source, grounded, rag_enabled, reground_applied, context_used, latency_ms,
retrieval`, and — when called with `include_sources=True` — `sources` (the
answerer's actual retrieved set, serialized with the `aiar.retrieve.v1` hit shape).

- `GET /calls/{call_id}` — compact, **redacted**, best-effort and process-local
  trace (date-partitioned JSONL with a FIFO cap; old ids `404`). `call_id` is the
  cross-route trace key.

## 5. Capability manifest — `GET /capabilities`

How a consumer decides what's available — derived from live predicates (mounted
**and** usable), never from the version string.

```jsonc
{
  "schema_version": "aiar.capabilities.v1",
  "aiar_version": "0.2.4", "backend_id": "...",
  "features": {
    "pure_retrieve": true, "remote_ingest": true, "grounding_v1": true,
    "semantic_grounding": false, "judge_only": false, "streaming": false,
    "answer_sources": true, "call_trace": true
  },
  "schemas": { "retrieve": "aiar.retrieve.v1", "ingest": "aiar.ingest.v1",
               "grounding": "aiar.grounding.v1", "answer": "aiar.answer.v1" }
}
```

`/healthz` additionally exposes `pure_retrieve`, `remote_ingest`,
`retrieve_schema_version`, `active_model`, and `active_model_ready`.

## 6. Active model — selection, readiness, and the `model_not_pulled` error

Generation calls resolve a model in this precedence:

1. explicit per-request `model` (e.g. `POST /services/prompt {"model": "..."}`),
2. the process-active model (runtime-settable, see below),
3. `OLLAMA_MODEL` env (the boot seed),
4. built-in default.

**Set it without a redeploy:** `POST /services/model {"model": "qwen3.5:9b"}`
(authed). It validates the target is pulled in Ollama and flips the active model
live; an unpulled target returns the structured 409 below. The Python equivalent
is `aiar.llm.set_active_model(name)`.

**Readiness is observable before a request fails.** `/healthz` and
`/services/meta` carry `active_model` + `active_model_ready: bool`, and
`/capabilities` carries `features.generation`. A box can be `ok` for
retrieval/ingest while `active_model_ready` is `false` because the configured
model isn't pulled — generation will fail until you `ollama pull` it or repoint.

**Unpulled model → structured, operator-fixable error** (a 4xx, distinct from the
transient `ollama_error` 503):

```jsonc
// 409
{ "detail": {
    "code": "model_not_pulled",
    "error": "model not pulled: 'qwen2.5:7b' (available: [...])",
    "model": "qwen2.5:7b",
    "available_models": ["gemma3:27b", "qwen3.5:9b", "nomic-embed-text"]
} }
```

**Optional auto-fallback:** set `AIAR_ACTIVE_MODEL_FALLBACK=auto` (default OFF) to
substitute the smallest pulled non-embedding model when the configured active
model is missing (logged loudly; embedding models are excluded). AIAR never
auto-`ollama pull`s — pulling a multi-GB model is the operator's call.

## Error `code` vocabulary

`unknown_instance` (404), `empty_query` (400), `unknown_job` (404),
`too_many_documents` (413), `payload_too_large` (413), `protected` (400),
`store_not_ready`/`embedder_unavailable` (503), `ollama_error` (503),
`unknown_call` (404). New codes are additive; existing codes never change meaning
within a version.
