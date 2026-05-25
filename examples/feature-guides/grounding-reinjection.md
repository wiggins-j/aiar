# Grounding reinjection

## What it is

When you score an answer low and write a **correction** (what it *should* have
said), AIAR stores that correction keyed to the prompt's signature. **Grounding
reinjection** automatically prepends the matching correction(s) to future answers
for that prompt — so a fix you make once *sticks* and the next answer is right.

This is the "reground" half of AIAR's loop: *ingest → retrieve → answer → judge →
correct → **reground** → verify.* Implementation:
[`aiar/grounding/`](../../aiar/grounding/); corrections persist as JSON under
`GROUNDING_BASE_DIR`. The watcher's **Evaluation queue** "Submit + Reground" writes
them; per-instance corrections never leak across instances.

## When it helps / when to skip

- **Helps** any time you want human corrections to persist without re-ingesting or
  retraining — it's how the assistant improves from your feedback over time.
- **Cost**: negligible (a lookup + a short prepend). Safe to leave on.

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `GROUNDING_REINJECTION_ENABLED` | `0` | auto-apply stored corrections to every answer |
| `GROUNDING_BASE_DIR` | `$HOME/.aiar` | where corrections are stored |

Note: even with the global flag off, the GUI **Reground** checkbox and the CLI
`--reground` flag force reinjection for a single call.

## Set it up

1. Enable always-on reinjection:
   ```bash
   export GROUNDING_REINJECTION_ENABLED=1
   ```
2. Create a correction to test it. **GUI:** on **Simulate** ask a prompt the model
   gets wrong, **Mark for evaluation**; on the **Evaluation queue** score it low,
   write the correct answer, **Submit + Reground**.

   **Headless / CLI (no browser):** there is no standalone grounding CLI, so use the
   watcher's JSON API (`python -m web.server`, then `curl` — no browser needed):
   ```bash
   # a) run the prompt and note the printed call_id (or use --json)
   python -m aiar.harness "the prompt the model gets wrong" --no-rag
   # b) queue that call, then submit the verdict + correction
   curl -s -X POST http://127.0.0.1:8088/api/activity/evaluate \
     -H 'Content-Type: application/json' -d '{"call_id":"<call_id>"}'
   curl -s -X POST http://127.0.0.1:8088/api/evaluation/verdict \
     -H 'Content-Type: application/json' \
     -d '{"call_id":"<call_id>","score":3,"correction":"what it should say"}'
   ```
3. Verify it sticks — re-ask the **same** prompt:
   ```bash
   python -m aiar.harness "the same prompt" --reground
   ```
   The answer should now reflect your correction (the GUI shows a green
   "Reground: applied" badge), and the judge verdict should rise.

## Tuning

- Corrections are matched by a normalized prompt signature, so near-identical
  phrasings reuse the same correction. Keep corrections concise and factual.
