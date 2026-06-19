<p align="center">
  <img src="web/static/aiar-logo.png" alt="AIAR logo" width="280" />
</p>

# AIAR — Local RAG with LLM-as-judge and a grounding loop

AIAR is a local-first retrieval-augmented generation framework for Python.
It runs against your own [Ollama](https://ollama.com) instance, ingests your
own documents, and ships three production-grade primitives out of the box:
hybrid retrieval, an LLM-as-judge that returns a structured `Verdict`, and a
grounding store that lets accepted corrections feed back into future answers.
It is built for developers, researchers, and AI hobbyists who want to own
their stack end to end — no cloud calls, no telemetry, no vendor lock-in.

## Install

```bash
pip install aiar-rag
# or, with the full retrieval extras (BM25, cross-encoder reranker, HyDE):
pip install 'aiar-rag[rag]'
```

> **Note on the name:** the distribution on PyPI is `aiar-rag` because `aiar`
> was already taken. The import package remains `aiar` — so your code uses
> `import aiar`, but you install with `pip install aiar-rag`.

**Prerequisites:** Python 3.10+ and a running [Ollama](https://ollama.com)
daemon (default `http://127.0.0.1:11434`). Pull at least one chat model and
one embedding model, for example:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

## Quickstart

```python
from aiar.harness.pipeline import answer_prompt

result = answer_prompt("What did our Q3 deployment doc say about rollback?", judge=True)
print(result["answer"])
print(result["verdict"])   # {"label": "Supported" | "Unsupported" | ..., "rationale": "...", ...}
```

`result` also carries `grounded`, `reground_applied`, `retrieval`, and
`latency_ms` so you can wire the loop into your own UI or pipeline.

## Why AIAR

Most local-RAG stacks stop at "retrieve and stuff into a prompt." AIAR
treats the answer as the *beginning* of the loop, not the end. The judge
catches hallucinations the moment they happen; the grounding store makes
sure the same hallucination does not happen twice. The whole system runs
on a laptop with a Qwen-class model and no external API calls — which
means you can ship it into environments where cloud calls are not allowed,
and you can audit every byte the model sees.

## The three wedges

### Hybrid retrieval

AIAR fuses lexical and semantic retrieval rather than picking one. Every
query runs through BM25 over a tokenized index *and* a vector search over
Ollama embeddings; the two ranked lists are merged with reciprocal-rank
fusion (RRF), then optionally reranked by a cross-encoder. HyDE-style query
rewriting and configurable `top_k` / `fetch_k` give you knobs without
forcing a tuning project on day one.

### LLM-as-judge Verdict

Every answer can be graded by a second LLM call that returns a structured
`Verdict`: a label (`Supported`, `Partially supported`, `Unsupported`,
`Off-topic`), a rationale, and the citations actually relied on. The judge
sees the same retrieved context as the answerer, so its critique is grounded
in evidence — and downstream code can branch on `verdict.label` to gate,
retry, or escalate.

### Grounding loop

When a Verdict is accepted (by a human, by automation, or by policy), the
answer plus its supporting context is persisted to a grounding store keyed
on the prompt. Next time a similar prompt arrives, AIAR reinjects that
grounding *before* the answerer runs and flags `reground_applied=True`. The
system stops re-making the same mistake — your corrections compound.

## What is in the box

- `aiar.harness.pipeline.answer_prompt` — the one-call entry point used
  in the Quickstart above.
- `aiar.rag` — hybrid retrieval, BM25 + vector + RRF, optional
  cross-encoder reranker, HyDE rewriting.
- `aiar.eval` — the LLM-as-judge with structured `Verdict` schema.
- `aiar.grounding` — accepted-correction store and reground pipeline.
- `aiar.harness.service` — optional FastAPI service exposing
  `/services/prompt` and `/services/meta` for other apps on the box.
- `aiar.observability` — call IDs, latency, and retrieval traces for
  every answer.

## Integration contracts (for apps built on AIAR)

AIAR exposes stable, schema-versioned contracts so a consuming app can use the
framework's truth (retrieval, grounding, ingest readiness, telemetry) instead of
reimplementing it. Each shape has **one serializer** shared by the in-process
Python API and the HTTP route, so local and remote callers get identical
payloads. See [docs/integration-contracts.md](docs/integration-contracts.md)
for the frozen field-level shapes.

- **Pure retrieval** (`aiar.retrieve.v1`) — `aiar.rag.retrieve_chunks(query, *,
  instance, k=8, category=None)` and `GET/POST /instances/{instance}/retrieve`.
  Raw vector similarity; **never invokes a generation model**.
- **Grounding** (`aiar.grounding.v1`) — `aiar.grounding.record_grounding(...)` /
  `lookup_grounding(...)`, keeping `answer` and `correction` distinct and scoping
  records per instance. Legacy records remain readable.
- **Ingest result + readiness** (`aiar.ingest.v1`) — `aiar.rag.ingest_documents(
  documents, *, instance, publish=False)`; `health(instance)` reports
  `published`, `chunk_count`, `last_ingest_at`, `last_ingest_error`.
- **Telemetry + capability manifest** — `answer_prompt(..., include_sources=True)`
  attaches the answerer's source set; `GET /capabilities` returns the feature
  manifest a consumer gates on (never a version string); `GET /calls/{call_id}`
  returns a redacted trace; `/healthz` carries `retrieve_schema_version`.
- **Active-model reliability** — `active_model_ready` on `/healthz` &
  `/services/meta` and `features.generation` on `/capabilities` flag when the
  configured model isn't pulled; generation then returns a structured
  `model_not_pulled` 4xx (not an opaque 503). Repoint live with authed
  `POST /services/model`, or opt into `AIAR_ACTIVE_MODEL_FALLBACK=auto`.

## Configuration

AIAR reads `AIAR_*` environment variables for runtime configuration —
endpoints, model names, reranker toggles, grounding-store paths, instance
isolation, and so on. Sensible defaults work out of the box for a local
Ollama install; see [PLAYBOOK.md](PLAYBOOK.md) for the full matrix.

## NEXT STEPS

Tune retrieval quality with the worked guides in `examples/feature-guides/`,
starting with
[improving-rag.md](examples/feature-guides/improving-rag.md).

## Contributing

PRs welcome. The deep-dive operator guide lives at
[PLAYBOOK.md](PLAYBOOK.md) — end-to-end walkthrough covering ingestion,
the harness, the watcher GUI, regrounding, evals, and operational notes.
Worked examples live under
[examples/feature-guides/improving-rag.md](examples/feature-guides/improving-rag.md).

For framework-level discussion, file an issue.

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), or the `license` field in
[pyproject.toml](pyproject.toml).
