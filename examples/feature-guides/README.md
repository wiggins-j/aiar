# Retrieval-quality feature guides

AIAR ships with a plain vector-search RAG by default, plus a set of **optional
quality methods** that you turn on to trade a little latency for a lot of
relevance. These are the same knobs you'd tune to make a RAG "work well."

Each guide below is **dual-purpose**:

- **Definition** — ask your AI agent *"What is the reranker?"* and it answers from
  the guide's **What it is** section.
- **Runnable setup** — say *"Let's set up the reranker"* and the agent opens the
  matching file and follows its **Set it up** steps (set env vars → first-run
  note → re-run the harness → measure the lift → persist the setting).

**New here?** Start with the roadmap — [`improving-rag.md`](improving-rag.md) — which
ranks every improvement by ROI (sources → parsing → metadata → chunking → retrieval
→ answer discipline → eval) and points to the per-method guides below.

| Method | File | Ask / say |
|---|---|---|
| **Roadmap: improving your RAG** | [`improving-rag.md`](improving-rag.md) | "How do I improve my RAG?" |
| Hybrid retrieval (vector + BM25) | [`hybrid-retrieval.md`](hybrid-retrieval.md) | "What is hybrid retrieval?" / "Set up hybrid retrieval" |
| Cross-encoder reranker | [`reranker.md`](reranker.md) | "What is the reranker?" / "Set up the reranker" |
| Query rewrite / HyDE | [`query-rewrite-hyde.md`](query-rewrite-hyde.md) | "What is HyDE?" / "Set up query rewrite" |
| Grounding reinjection | [`grounding-reinjection.md`](grounding-reinjection.md) | "What is grounding reinjection?" / "Set up grounding" |
| Top-K (context size) | [`top-k.md`](top-k.md) | "What is top-k?" / "Tune top-k" |
| Measure the lift (A/B + ablation) | [`measure-lift.md`](measure-lift.md) | "How do I measure RAG quality?" |

## How to use these

1. Pick a method (or ask "what's recommended?" — start with the reranker + hybrid).
2. Read its **What it is** to decide if it fits your corpus and hardware.
3. Follow its **Set it up** steps. Every method is a single env var (plus a tuning
   var or two); nothing is code.
4. **Measure** with [`measure-lift.md`](measure-lift.md) so you keep what helps and
   drop what doesn't on *your* data.

All env vars are also documented in [`../../config.example`](../../config.example),
and the recommended-on defaults are offered during the setup prompt (see the repo
README, STEP 0 and the NEXT STEPS section).

> Order of the pipeline: `query → (rewrite/HyDE) → retrieve (hybrid | vector) →
> (rerank) → top-K → context`. Each stage is independent and off by default.

**CLI / headless note.** Every method here is command-line first: each is an env var
(set it in your shell or `config.example`), and you verify with `python -m
aiar.harness` and `python -m aiar.eval.runner`. The only browser-optional step —
recording a grounding correction — also has a `curl` path via the watcher API (see
[`grounding-reinjection.md`](grounding-reinjection.md)), so an SSH-only user can do
everything without a GUI.
