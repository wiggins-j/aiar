# Measure the lift (A/B + ablation)

Don't trust a quality knob on faith — **measure** what each one does on *your*
corpus and hardware, then keep what helps and drop what doesn't.

## The A/B runner

AIAR ships an evaluation runner that scores a set of cases with RAG **on** vs
**off** and reports the delta:

```bash
python -m aiar.eval.runner ./examples/cases.json
# reports: RAG ON x/N, RAG OFF x/N, DELTA (positive = RAG helped)
```

Point it at your own cases file (same shape as
[`examples/cases.json`](../cases.json)) once you have real questions for your corpus.
Each brief's "Example RAG Questions" are good seeds.

## Ablation — measure ONE feature at a time

The runner reports overall RAG lift, not per-feature. To see what a single method
contributes, **toggle that one flag, re-run, and compare the RAG-ON number** —
holding everything else constant:

```bash
# baseline with the feature OFF
RAG_RERANK_ENABLED=0 python -m aiar.eval.runner ./examples/cases.json
# same cases with the feature ON
RAG_RERANK_ENABLED=1 python -m aiar.eval.runner ./examples/cases.json
# keep it if the RAG-ON score went up enough to justify the latency
```

Repeat for `RAG_HYBRID_ENABLED`, `RAG_QUERY_REWRITE_MODE` (off/rewrite/hyde), and
`RAG_TOP_K` values. Change **one variable per run** so you can attribute the change.

## In the GUI

For a quick qualitative check without a cases file: on the **Simulate** page run the
same prompt with "Use RAG" on vs off (or switch the active instance to **No RAG** in
**Settings**), and watch the judge verdict and the retrieved context. The
**Activity** page shows the exact query/context used, which is handy when testing
query-rewrite/HyDE.

## Rule of thumb

Start from the recommended stack (hybrid + reranker + HyDE + grounding), then ablate
to confirm each earns its cost on your data. On weak/CPU-only hardware, reranker and
HyDE are the first to drop if latency hurts and the lift is small.
