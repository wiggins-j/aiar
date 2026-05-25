# Hybrid retrieval (vector + BM25)

## What it is

Dense **vector** search matches on *meaning* but can miss exact tokens (a part
number, an error code, a rare proper noun). **BM25** is classic keyword search that
nails exact terms but misses paraphrase. **Hybrid retrieval** runs both and merges
their rankings with **Reciprocal Rank Fusion (RRF)**, so you get semantic recall
*and* exact-term precision.

In AIAR, when enabled, retrieval fetches `RAG_VECTOR_K` vector hits and `RAG_BM25_K`
BM25 hits, then RRF-fuses them (`RAG_RRF_K`) to a single ranked list. Implementation:
[`aiar/rag/lexical.py`](../../aiar/rag/lexical.py) (BM25, built in-memory from the
instance's chunks) + [`aiar/rag/fusion.py`](../../aiar/rag/fusion.py) (RRF).

## When it helps / when to skip

- **Helps** when queries contain exact identifiers, codes, names, or jargon, or when
  pure vector search returns "close but not it" results.
- **Cost**: small — the BM25 index is built in memory from chunks already in the
  store (no second copy on disk), and fusion is cheap. Good default to leave on.

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `RAG_HYBRID_ENABLED` | `0` | turn hybrid fusion on |
| `RAG_VECTOR_K` | `20` | vector candidates fetched before fusion |
| `RAG_BM25_K` | `20` | BM25 candidates fetched before fusion |
| `RAG_RRF_K` | `60` | RRF constant (higher = flatter rank weighting) |

## Set it up

1. Enable it:
   ```bash
   export RAG_HYBRID_ENABLED=1
   ```
2. Confirm it works — exact-term queries should now surface the right chunk:
   ```bash
   python -m aiar.harness "a query with an exact term/code from your docs"
   ```
3. **Measure the lift** — see [`measure-lift.md`](measure-lift.md): A/B with
   `RAG_HYBRID_ENABLED=0` vs `=1` and compare.

## Tuning

- This is the ideal **first pass for the reranker** — turn both on together
  (hybrid recall → reranker precision). See [`reranker.md`](reranker.md).
- Raise `RAG_VECTOR_K`/`RAG_BM25_K` for more recall before fusion.
