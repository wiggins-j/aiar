# Improving your RAG — a roadmap after the first build

You've built an initial RAG (ingested docs, asked questions, seen grounded answers).
This is the roadmap for making it *good* and keeping it good — ordered by **return
on investment**, with each step mapped to what AIAR already gives you, what's easy
to add, and what's advanced/external.

> **The one principle:** fix **sources → parsing → metadata → chunking → retrieval
> → answer discipline → evaluation** *before* reaching for fancy methods. Most
> quality problems are upstream of the vector database. Measure every change.

> **Small-local-model lens:** AIAR's whole point is a free local model competing
> with frontier models on *your* docs. Every extra LLM call (query rewrite, HyDE,
> multi-query, verification, agents) costs real latency on a local box. Prefer the
> cheap, high-ROI levers first, and **measure** whether each pricier step earns its
> cost on your hardware ([`measure-lift.md`](measure-lift.md)).

---

## The default stack (start here)

```
good sources → clean parsing → trust metadata → smart chunking →
hybrid search → reranker → cite-and-refuse answers → eval loop
```

That pipeline beats most "fancy" RAG setups. Everything below is how to build it and
where to go after.

---

## Tier A — Foundations (highest ROI, do first)

These pay off more than any model or vector-DB choice.

1. **Source quality + trust tiers.** Classify every document by authority, date,
   domain, and claim type *before* embedding, so retrieval and the answer can prefer
   authoritative chunks (official label > guideline > trial > observational >
   adverse-event > anecdote). **In AIAR:** the collection brief defines the tiers and
   the **metadata-schema framework** enforces them —
   `python -m aiar.rag.metadata validate <corpus> --schema <s.json>` (see
   [`../metadata-schemas/`](../metadata-schemas/)). Bad metadata is worse than none,
   so validate.
2. **An evaluation set.** Write test questions with known-good answers/sources and
   score every change against them — did it retrieve the right chunk, answer
   correctly, cite the right source, refuse when evidence was thin. **In AIAR:**
   `python -m aiar.eval.runner ./examples/cases.json` (RAG on/off lift); add your own
   cases. Without this you're tuning on vibes.
3. **Clean parsing.** Garbage in, garbage out — extract tables, headings, lists, and
   footnotes cleanly before embedding. Often a bigger win than swapping embedding
   models. **In AIAR:** ingest text/markdown/JSON; pre-convert PDFs/HTML to clean
   `.md`/`.txt` with a tool of your choice first.
4. **Smart chunking.** Split on structure (heading → section → paragraph), keep a
   topic per chunk, and carry the title/source into each chunk. No universal size —
   too small loses context, too big retrieves poorly. Measure a few sizes.

## Tier B — Retrieval upgrades AIAR already ships

Turn these on and measure; they're the biggest retrieval wins. Each has a guide:

5. **Hybrid search** (vector + BM25, RRF-fused) — exact terms *and* meaning.
   [`hybrid-retrieval.md`](hybrid-retrieval.md)
6. **Reranker** (cross-encoder over a wide first pass) — the best precision upgrade.
   [`reranker.md`](reranker.md)
7. **Query rewrite / HyDE** — closes the vocabulary gap.
   [`query-rewrite-hyde.md`](query-rewrite-hyde.md)
8. **Top-K tuning** — smallest context that answers correctly.
   [`top-k.md`](top-k.md)
9. **Multi-index routing** — keep separate corpora and switch per question. **In
   AIAR:** named **RAG instances** (Settings → Active RAG Instance, or `--instance`);
   the briefs ingest each source group into its own instance.

## Tier C — Answer discipline (trust + safety)

10. **Cite-and-refuse generation.** Answer only from retrieved chunks, cite each
    claim, prefer higher-authority sources, and say "I don't have enough evidence"
    rather than guess. **In AIAR:** put this contract in the **system prompt**
    (Settings → system-prompt editor; save reusable ones as presets). The briefs'
    "Assistant Response Format" is written for exactly this.
11. **The judge (groundedness check).** Score answers for quality/faithfulness so
    regressions are visible. **In AIAR:** the built-in LLM-as-judge runs on every
    harness/Simulate call (toggle it on the Simulate page).
12. **The reground loop (human feedback that sticks).** Correct a wrong answer once
    and have it persist — AIAR's signature feature and a very high-ROI, low-cost
    quality lever. **In AIAR:** [`grounding-reinjection.md`](grounding-reinjection.md)
    (Evaluation queue → Submit + Reground).
13. **No-answer thresholds.** Refuse when the best chunk is too weak — big
    hallucination reduction for medical/legal/safety domains. **Not built into AIAR
    yet** — today the judge flags weak answers after the fact; a retrieval-score
    threshold is a worthwhile addition to explore.

## Tier D — Add when you outgrow the basics

Worth it once Tiers A–C are solid and your eval set shows a specific gap. Several
add an extra LLM call — measure the latency on your local model.

- **Multi-query + Reciprocal Rank Fusion** — generate query variations, retrieve
  each, fuse. (AIAR already uses RRF inside hybrid search; multi-query is the
  not-yet-built extension.)
- **Parent-document / hierarchical (auto-merging) retrieval** — embed small chunks,
  return the larger parent section. Great for manuals and long docs. *(Explore.)*
- **Contextual retrieval** — prepend a short "where this chunk sits" header before
  embedding (Anthropic's technique). Helps tiny chunks. *(Explore.)*
- **Metadata filtering / self-query** — filter by year/model/region/source before
  semantic search. AIAR supports a metadata `where=` filter and `--category` tags;
  LLM-written self-query filters are the extension.
- **Recency / version-aware retrieval** — prefer the newest label/guideline; keep old
  versions for historical questions. Your schema's `published_date`/`last_updated`
  fields enable this.
- **Deduplication & canonicalization** — drop near-duplicate chunks (same recall on
  five sites, HTML+PDF of one doc) so outdated copies don't win.
- **Source-conflict detection** — when sources disagree, surface it and prefer the
  higher tier (your trust metadata makes this possible).
- **Query decomposition** — split comparison/multi-hop questions into sub-questions.
- **Context compression** — extract only the relevant span of each chunk before
  sending to the model (saves the local model's context window).

## Tier E — Advanced / external (only after the basics + eval prove a bottleneck)

These are powerful but heavy, and mostly **outside** AIAR's built-in scope. Reach for
them only when your eval set shows a ceiling the simpler tiers can't break:

- **Domain-specific or fine-tuned embeddings / fine-tuned reranker** — needs eval
  data to prove a gain; can regress on general language.
- **GraphRAG / knowledge graph** — great for relationship-heavy, multi-hop corpora;
  expensive to build and maintain; overkill for FAQ/manual RAG.
- **Late-interaction (ColBERT) / learned sparse (SPLADE)** — stronger retrieval,
  more infra; usually unnecessary for a first production RAG.
- **Multimodal retrieval** — only if your corpus is genuinely visual (diagrams,
  scans); high hallucination risk and hard to cite.
- **Agentic / iterative (corrective) RAG, long-context dumping, autonomous
  self-improvement** — useful for research agents; slower, less deterministic, more
  failure modes. Not where a knowledge-base Q&A system should start.
- **Generator fine-tuning / distillation into a smaller model** — improves *style*
  and format more than factuality; bad retrieval still yields bad answers.

The wider ecosystem (RAGAS/TruLens for eval, LlamaIndex/LangChain/Haystack for
retrieval patterns, Qdrant/Weaviate/Milvus/Elasticsearch for scale) is worth knowing
when you outgrow a single-box setup — but you do **not** need any of it to get a
strong AIAR RAG first.

---

## The loop that actually improves things

1. Pick **one** change from the highest tier you haven't finished.
2. Re-run your eval set ([`measure-lift.md`](measure-lift.md)) — one variable at a time.
3. Keep it only if the number improves *and* the latency is acceptable on your model.
4. Repeat. Most teams skip straight to "which vector DB / embedding / agent
   framework?" — the real questions are: *are my sources trustworthy, my chunks
   clean, can I measure retrieval, and can I prove the answer came from the right
   source?* Once those are true, rerankers, hybrid search, and the rest pay off.

> Method descriptions here are general RAG practice, framed for AIAR. The per-method
> guides in this folder are the runnable, AIAR-specific recipes.
