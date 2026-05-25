# Cross-encoder reranker

## What it is

Plain vector search ranks chunks by embedding similarity — fast, but it judges the
query and each chunk *separately*. A **cross-encoder reranker** re-scores the top
candidates by feeding the **query and chunk together** through a small relevance
model, which is much better at telling "actually answers this" from "merely related."

In AIAR the pipeline does a **wide first pass** (fetch `RAG_FETCH_K` candidates via
vector or hybrid retrieval), then the reranker re-orders them and keeps the best
`RAG_TOP_K`. Implementation: [`aiar/rag/reranker.py`](../../aiar/rag/reranker.py),
wired in [`aiar/rag/retriever.py`](../../aiar/rag/retriever.py).

## When it helps / when to skip

- **Helps** most when your corpus is large or chunks are similar-sounding, and you
  need the *single most correct* passage near the top.
- **Cost**: the first run downloads the cross-encoder model
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80 MB) and each query adds a rerank
  pass. On weak/CPU-only boxes this adds noticeable latency — measure before
  committing.

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `RAG_RERANK_ENABLED` | `0` | turn the reranker on |
| `RAG_FETCH_K` | `20` | how many candidates the wide first pass fetches to rerank |
| `RAG_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | the cross-encoder |

## Set it up

1. Enable it (add to your shell profile or `config.example`, then re-`source`):
   ```bash
   export RAG_RERANK_ENABLED=1
   export RAG_FETCH_K=20
   ```
2. **First run downloads the model** — trigger it once so the download happens now:
   ```bash
   python -m aiar.harness "a question your docs answer" --top-k 3
   ```
   (It will be slow the first time, then cached under your HuggingFace cache.)
3. Confirm it's active: the harness/GUI answers should now pull more on-point
   chunks. In the watcher GUI, re-run the same prompt on **Simulate** and compare.
4. **Measure the lift** — see [`measure-lift.md`](measure-lift.md): run the A/B
   runner with `RAG_RERANK_ENABLED=0`, then `=1`, and compare the RAG-on score.
5. Keep it if the delta is positive *and* the latency is acceptable for your box.

## Tuning

- Raise `RAG_FETCH_K` (e.g. 30–50) to give the reranker more candidates (slower,
  sometimes better recall). Lower it on weak hardware.
- Pairs well with **hybrid retrieval** as the first pass — see
  [`hybrid-retrieval.md`](hybrid-retrieval.md).
