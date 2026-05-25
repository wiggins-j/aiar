# Query rewrite / HyDE

## What it is

The user's wording often doesn't match how the answer is written in your docs —
the "vocabulary gap." Two pre-retrieval techniques close it:

- **rewrite** — the LLM rewrites the question into a cleaner search query (expands
  abbreviations, adds domain terms) before retrieval.
- **HyDE** (Hypothetical Document Embeddings) — the LLM writes a short *hypothetical
  answer paragraph*, and AIAR retrieves against **that** (an answer looks more like
  the target chunk than a question does), which usually retrieves better.

Set the mode with `RAG_QUERY_REWRITE_MODE`. Implementation:
[`aiar/rag/query_rewrite.py`](../../aiar/rag/query_rewrite.py). Each named RAG
instance can carry its own rewrite/HyDE prompts (the Tesla/OSRS examples do), or
fall back to generic built-ins.

## When it helps / when to skip

- **Helps** when users ask in plain language but docs are technical, or terminology
  differs from everyday phrasing. **HyDE** is the strongest of the two.
- **Cost**: one extra LLM call per query before retrieval (latency + tokens). On a
  slow model this is the most noticeable knob — measure.

## Env vars

| Var | Default | Values |
|---|---|---|
| `RAG_QUERY_REWRITE_MODE` | `off` | `off` \| `rewrite` \| `hyde` |

## Set it up

1. Pick a mode (start with `hyde`):
   ```bash
   export RAG_QUERY_REWRITE_MODE=hyde
   ```
2. Confirm it works — paraphrased questions should now retrieve the right chunk:
   ```bash
   python -m aiar.harness "ask in casual words something your docs cover formally"
   ```
   Use `--think` to see the model's reasoning, or check the **Activity** page in the
   GUI to see the rewritten/HyDE query that was actually retrieved against.
3. **Measure the lift** — see [`measure-lift.md`](measure-lift.md): compare
   `off` vs `rewrite` vs `hyde` on your cases and keep the best.

## Tuning

- Per-instance prompts: set `query_rewrite.rewrite_system` / `hyde_system` on a RAG
  instance's registry descriptor to make the rewrite domain-aware (operator config,
  never hard-coded domain text in the framework).
- If latency matters more than recall, try `rewrite` (cheaper) before `hyde`.
