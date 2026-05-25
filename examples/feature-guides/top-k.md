# Top-K (context size)

## What it is

`RAG_TOP_K` is how many retrieved chunks get injected into the prompt as context.
More chunks = more chance the answer is present, but also more noise and tokens
(and, past a point, worse answers as the model gets distracted).

It's the last stage of the pipeline: after retrieval (and any reranking), the top
`RAG_TOP_K` chunks become the labelled `--- Knowledge ---` block. See
[`aiar/rag/retriever.py`](../../aiar/rag/retriever.py).

## When to change it

- **Raise** (e.g. 5–8) if answers miss facts that *are* in your corpus — the right
  chunk may be ranked just outside the window.
- **Lower** (e.g. 2) if answers ramble or pull in irrelevant context, or to save
  tokens/latency on a small model.

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `RAG_TOP_K` | `3` | chunks injected into each prompt |

## Set it up

1. Set it:
   ```bash
   export RAG_TOP_K=5
   ```
   Or override per call without changing the env: `python -m aiar.harness "..." --top-k 5`.
2. **Measure** — see [`measure-lift.md`](measure-lift.md): try a few values on your
   cases and keep the smallest `RAG_TOP_K` that answers correctly (smaller is
   cheaper and less noisy).

## Tuning

- With the **reranker** on, a small `RAG_TOP_K` (3) is usually enough because the
  best chunks are already at the top — let `RAG_FETCH_K` do the wide search and
  `RAG_TOP_K` stay tight. See [`reranker.md`](reranker.md).
