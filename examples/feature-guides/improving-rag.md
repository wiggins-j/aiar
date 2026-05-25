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

### The next enhancements to consider

These are the highest-value *future* frameworks to add to AIAR after the current
stack (hybrid + rerank + rewrite/HyDE + judge + reground loop) is in good shape.
Each item below is framed by **priority**, **when to use it**, and **when not to
use it** so teams don't cargo-cult whatever was in the latest RAG blog post.

1. **Contextual retrieval** *(Priority: High)* — generate a short chunk-specific
   context from the full document, then prepend it before embedding **and** before
   building the BM25 index (Anthropic's "Contextual Retrieval" idea). This is one
   of the best upgrades once your chunking is already clean.
   - **Use it when:** chunks are individually correct but too local to retrieve well;
     you have long documents (manuals, labels, policies, papers); neighboring chunks
     need surrounding section meaning to rank correctly.
   - **Do not use it when:** the corpus is already short/self-contained; ingest cost
     matters more than recall; the local box cannot afford an extra preprocessing
     pass over every chunk.
   - **AIAR status:** not built yet; recommended future enhancement.
2. **Metadata filtering / self-query retrieval** *(Priority: High)* — use trusted
   metadata (year/model/region/source tier/claim type/product/version) to filter
   or route retrieval before semantic search.
   - **Use it when:** the corpus has strong front-matter; different source classes
     should answer different questions; recency, geography, product version, or
     authority level matter.
   - **Do not use it when:** metadata quality is weak or inconsistent; the corpus is
     tiny; you would be forcing brittle filters onto naturally fuzzy questions.
   - **AIAR status:** partial. AIAR stores metadata and supports `where=` filters;
     LLM-written self-query and automatic routing are the extension.
3. **Parent-document / hierarchical retrieval** *(Priority: High)* — retrieve with
   small chunks, but expand the final context to the larger parent section or
   document span.
   - **Use it when:** users ask questions that need a few surrounding paragraphs,
     table notes, section headers, contraindications, definitions, or caveats.
   - **Do not use it when:** context windows are very tight; the corpus is mostly
     short FAQ-style entries; expanding to parents would just add noise.
   - **AIAR status:** not built yet; recommended future enhancement.
4. **Recency / version-aware retrieval** *(Priority: High in regulated or changing
   domains; Medium otherwise)* — bias retrieval and answer synthesis toward newer
   documents, while preserving access to older versions for historical questions.
   - **Use it when:** docs change over time; policy, medical, legal, release-note,
     product-manual, or pricing questions depend on version/date.
   - **Do not use it when:** the corpus is static or timeless; freshness is less
     important than canonical authority.
   - **AIAR status:** partial. The metadata framework supports this; retrieval logic
     does not yet use dates as a first-class ranking feature.
5. **Source-conflict detection + authority-aware answering** *(Priority: High in
   health/legal/safety; Medium elsewhere)* — explicitly surface when sources
   disagree and prefer the higher-trust tier.
   - **Use it when:** the corpus intentionally mixes labels, guidelines, studies,
     reviews, reports, and weaker evidence; users need to know disagreement exists.
   - **Do not use it when:** the corpus is already canonical and internally uniform;
     conflict logic would add complexity without user value.
   - **AIAR status:** partial. The metadata scaffolding exists; automated conflict
     detection and authority-aware ranking are future work.
6. **Eval rigor upgrades** *(Priority: High)* — expand the current eval loop into a
   stronger benchmark system: fixed golden sets, richer retrieval/groundedness
   metrics, factorized judge scoring, and saved experiment comparisons. This is the
   operational lesson from the Google and Databricks evaluation posts.
   - **Use it when:** multiple people are tuning the RAG; you are changing chunking,
     prompts, models, or retrieval knobs often; regressions matter.
   - **Do not use it when:** you're still at "does this even work on my docs?"
     prototype stage and have not yet written a meaningful case set.
   - **AIAR status:** partial. AIAR already has case-file A/B eval and an LLM judge;
     benchmark management and richer metrics are the extension.
7. **Multi-query retrieval** *(Priority: Medium)* — generate multiple query
   variants, retrieve on each, then fuse results. AIAR already uses RRF inside
   hybrid search; this is the query-side extension.
   - **Use it when:** user wording is highly variable; the same concept appears under
     many names; recall is the bottleneck even after rewrite/HyDE.
   - **Do not use it when:** rerank + hybrid + rewrite already solve the problem; an
     extra LLM call per query is too expensive on the local model.
   - **AIAR status:** not built yet; worth exploring after contextual retrieval and
     metadata-aware retrieval.
8. **Deduplication & canonicalization** *(Priority: Medium)* — collapse near-duplicate
   content and designate canonical sources so stale mirrors do not win retrieval.
   - **Use it when:** the same content exists as PDF + HTML + copied summaries; many
     near-identical versions pollute recall.
   - **Do not use it when:** the corpus is already curated and version-controlled;
     duplicates are rare and easy to inspect manually.
   - **AIAR status:** not built yet; useful but usually lower ROI than metadata and
     contextual retrieval.
9. **Context compression / span extraction** *(Priority: Medium)* — once retrieval is
   good, compress the returned context to only the answer-relevant spans before
   sending it to the model.
   - **Use it when:** the local model has limited context budget; retrieved chunks are
     long; parent retrieval makes answers better but too verbose.
   - **Do not use it when:** the answer model already handles the current context
     volume comfortably; compression would add another failure point.
   - **AIAR status:** not built yet; explore after parent retrieval exists.
10. **Query decomposition** *(Priority: Medium)* — split comparison or multi-hop
    questions into sub-questions, retrieve each separately, then synthesize.
    - **Use it when:** users ask "compare X vs Y", "what changed between versions",
      or questions that need evidence from multiple documents.
    - **Do not use it when:** most traffic is single-hop lookup; the extra query
      orchestration adds more latency than value.
    - **AIAR status:** not built yet; useful for specific corpora, not a default.

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

### Frameworks that are interesting but usually not the next move

The following ideas are real, but they are easy to over-apply:

- **Multi-agent / orchestrator-worker research systems** *(Priority: Low for AIAR's
  current mission)* — inspired by Anthropic's multi-agent research architecture.
  - **Use it when:** the task is open-ended research over many tools/sources, not
    just grounded Q&A over a known corpus.
  - **Do not use it when:** the product is still a document-grounded assistant; you
    have not yet maxed out static retrieval quality; reproducibility matters more
    than exploratory breadth.
  - **AIAR status:** out of scope for the core RAG path right now.
- **GraphRAG / knowledge graph** *(Priority: Low unless the corpus is relationship
  heavy)*.
  - **Use it when:** users ask genuine entity-relationship and multi-hop questions
    over dense, structured facts.
  - **Do not use it when:** the corpus is FAQ/manual/guideline heavy and standard
    retrieval already works; graph construction would be expensive ceremony.
  - **AIAR status:** not built; intentionally deferred.
- **Late-interaction retrieval (ColBERT) / learned sparse retrieval (SPLADE)**
  *(Priority: Low)*.
  - **Use it when:** eval proves a real retrieval ceiling that hybrid + rerank +
    contextual retrieval cannot break.
  - **Do not use it when:** you are still in single-node local-RAG territory; the
    operational complexity outweighs the gain.
  - **AIAR status:** not built; external/advanced.

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
