# AIAR Playbook — end to end

A complete, copy-paste walkthrough: pull a Qwen model, ingest your own
documents, run the harness, turn on the reranker + grounding, launch the GUI,
simulate a prompt, evaluate it, **reground**, and verify the regrounded answer
improved. Real commands, real paths.

Every shell line below assumes you are in the repo root (wherever you cloned
AIAR, e.g. `~/aiar`). AIAR is **OS-agnostic** — Linux (incl. Ubuntu LTS
servers), macOS, and Windows. Shell snippets use Linux/macOS syntax; Windows
equivalents are called out where they differ (venv activation, `export` →
`$env:`, reset). If your machine does not expose bare `python`, substitute the
launcher it does provide (`python3` on many macOS/Linux systems, `py` on
Windows).

---

## 0. Prerequisites

- **Any OS** — Linux (incl. headless Ubuntu LTS servers), macOS, or Windows.
- Python 3.10–3.14. On newer active versions, the RAG stack may need to build
  `chroma-hnswlib` from source on some platforms if a wheel is unavailable.
- Disk/RAM/VRAM sized to the Qwen model you pick (see the table in §1). A
  mid-size 7–8B model needs ~5 GB disk and ~8 GB RAM (or ~6 GB VRAM).
- **Headless server note:** the CLI/A-B runner need no GUI. For the watcher GUI
  on a remote box, SSH-forward the port (`ssh -L 8088:127.0.0.1:8088 me@server`)
  or set `AIAR_WEB_HOST=0.0.0.0` behind a firewall/tunnel.

---

## 1. Install Ollama and pull any Qwen model

AIAR is **model-agnostic** — it works with any Qwen model you have pulled.

```bash
# Install Ollama (see https://ollama.com/download for your OS)
#   macOS:    brew install ollama   (or the .dmg)
#   Linux:    curl -fsSL https://ollama.com/install.sh | sh
#   Windows:  run the installer from https://ollama.com/download (it runs as a service)

# Start the Ollama server (leave running; or use the menubar/tray app / Windows service)
ollama serve &

# Pull a Qwen model sized to your hardware (see the table below). Examples:
ollama pull qwen2.5:7b                  # balanced default
#   ollama pull qwen2.5:3b              # smaller / faster, great for laptops
#   ollama pull qwen3:14b              # larger / slower / stronger

# Sanity check
ollama list
```

**Which Qwen?** Pick the largest that fits your machine (rough Q4_K_M-quantized
guidance — CPU-only works at every size, just slower):

| Params | Example Ollama tag | RAM (CPU) | VRAM (GPU) | Good for |
|---|---|---|---|---|
| ~1.5B | `qwen2.5:1.5b` | ~2 GB | ~2 GB | tiny boxes, smoke tests |
| ~3B | `qwen2.5:3b` | ~4 GB | ~4 GB | laptops, fast iteration |
| ~7–8B | `qwen2.5:7b`, `qwen3:8b` | ~8 GB | ~6 GB | **balanced default** |
| ~14B | `qwen2.5:14b`, `qwen3:14b` | ~16 GB | ~12 GB | stronger reasoning |
| ~32B | `qwen2.5:32b`, `qwen3:32b` | ~32 GB | ~24 GB | workstation / server |
| ~72B | `qwen2.5:72b` | ~48 GB+ | 2×24 GB | multi-GPU server |

> ⚠️ **Always check the canonical sources** — this table is a snapshot and will
> age, and the exact tag must exist for your Ollama version:
> - Qwen model cards & specs: <https://huggingface.co/Qwen>
> - Qwen 3.5 collection: <https://huggingface.co/collections/Qwen/qwen35>
> - Qwen 3.6 collection: <https://huggingface.co/collections/Qwen/qwen36>
> - Ollama-pullable tags: <https://ollama.com/library/qwen>
>
> Not sure of your specs? Detect them: Linux `free -h` + `nvidia-smi`; macOS
> `sysctl hw.memsize` (Apple Silicon: unified memory = your RAM); Windows
> `systeminfo` + `nvidia-smi`. For a remote box, prefix with `ssh me@host '...'`.

---

## 2. Install AIAR

```bash
git clone https://github.com/wiggins-j/aiar.git aiar && cd aiar
# or clone your own fork instead, or use an existing local checkout directly

python -m venv .venv
source .venv/bin/activate              # Windows (PowerShell): .venv\Scripts\Activate.ps1
                                       # Windows (cmd):        .venv\Scripts\activate.bat

# Core + the RAG stack (vector store, embeddings, BM25, reranker)
pip install -r requirements.txt -r requirements-rag.txt
```

Point AIAR at the model you pulled (this is the ONLY model setting):

```bash
export OLLAMA_MODEL="qwen2.5:7b"       # whatever you pulled in step 1
#   Windows (PowerShell):  $env:OLLAMA_MODEL="qwen2.5:7b"
```

Optional: source the full config template so every path/flag has a value:

```bash
source config.example      # then edit the exports to taste
```

---

## 3. Ingest YOUR documents into a RAG

Put your documents in a folder. Supported types: `.txt`, `.md`, `.markdown`,
`.rst`, `.json`. (AIAR ships a tiny example corpus in `examples/docs`.)

```bash
# Preview what would be ingested (no writes):
python -m aiar.rag.ingest ./examples/docs --dry-run

# Ingest for real (writes embeddings to ~/.aiar/knowledge):
python -m aiar.rag.ingest ./examples/docs

# Ingest your own folder, into a named RAG instance:
python -m aiar.rag.ingest /path/to/my/docs --instance mydocs --category mydocs
```

The store persists at `~/.aiar/knowledge` (override with `AIAR_DB_PATH`). Docs
land in the **active** RAG instance unless you pass `--instance <name>` (a new
instance is created on first ingest; the default instance's collection name is
`AIAR_CORPUS`, default `aiar`). In the GUI, that built-in `default` instance is
shown as **Example RAG**; CLI/API still use the slug `default`. To re-ingest
from scratch, delete the store
directory (`rm -rf ~/.aiar/knowledge`) and run ingest again.

---

## 3.5 (Optional) Let an AI build the corpus for you

Don't want to gather documents by hand? Hand an AI agent (one with web + file
tools) a **Collection Brief** — a single Markdown file that tells it which official
sources to use, what's in/out of scope, how to split + tag files, and which safety
rules to respect. The agent fetches + normalizes documents into a folder; you then
ingest that folder exactly as in §3.

Two common paths:

- **Full Tesla demo:** start from
  `examples/corpus-briefs/tesla-manual-expert-collection-brief.md`, collect the
  corpus into `corpus/tesla/`, then ingest with
  `python -m aiar.rag.ingest corpus/tesla --instance tesla`.
- **Your own domain:** start from
  `examples/corpus-briefs/collection-brief-builder-prompt.md`, let the AI write
  `briefs/<name>-collection-brief.md`, collect docs into `corpus/<name>/`, then
  ingest with `python -m aiar.rag.ingest corpus/<name> --instance <name>`.

```bash
# 1. Author your brief — copy the worked example and edit it...
mkdir -p briefs
cp examples/corpus-briefs/tesla-manual-expert-collection-brief.md briefs/mydomain-collection-brief.md
#    ...OR generate one: paste examples/corpus-briefs/collection-brief-builder-prompt.md
#       into your AI; it interviews you and writes the brief. Save it under briefs/.

# 2. Hand briefs/mydomain-collection-brief.md to an AI collector agent. It writes
#    clean .md/.txt/.json files (one topic/procedure each) into corpus/mydomain/.
#    (corpus/ is git-ignored — large + regenerable.)

# 3. Ingest the collected folder into its own instance:
python -m aiar.rag.ingest corpus/mydomain --instance mydomain --category mydomain
```

Notes:
- Keep the brief in `briefs/` (NOT inside `corpus/<name>/`) so it isn't indexed.
- The brief's "Assistant Response Format" → paste into the **system prompt**
  (Settings page, §6). Its example questions → `examples/cases.json` for the A/B
  runner (§4). See `examples/corpus-briefs/README.md` for the full workflow.

---

## 4. Run the harness against a prompt

```bash
python -m aiar.harness "How many days do I have to request a refund?"
```

You'll see the **answer**, then the **judge verdict** (rating + reason +
confidence), plus the `call_id` of the call (you'll use it in the GUI). The
harness pipeline is: `prompt → retrieve → (reground) → answer → judge`.

Useful flags:

```bash
python -m aiar.harness "..." --no-rag     # blind the answerer (no retrieval) — A/B baseline
python -m aiar.harness "..." --think      # show the model's step-by-step reasoning
python -m aiar.harness "..." --json       # full machine-readable result
```

**See retrieval lift with the A/B runner** (runs each case RAG-on AND RAG-off
and reports the delta):

```bash
python -m aiar.eval.runner ./examples/cases.json
# => DELTA : rubric +N  pct +XX%  (positive = RAG helped)
```

---

## 5. Enable the reranker + grounding flags

All retrieval-quality features default OFF (the bare path is plain vector
search). Turn them on for relevance:

```bash
# Hybrid retrieval: fuse dense vectors with sparse BM25 (great for exact terms)
export RAG_HYBRID_ENABLED=1

# Cross-encoder reranking: re-score a wide first pass, keep the best few
export RAG_RERANK_ENABLED=1
export RAG_FETCH_K=20            # how wide the first pass is before reranking

# Pre-retrieval query rewrite / HyDE (closes the vocabulary gap)
export RAG_QUERY_REWRITE_MODE=hyde     # off | rewrite | hyde

# Grounding reinjection: auto-apply past corrections to EVERY answer
export GROUNDING_REINJECTION_ENABLED=1
```

The first reranked query downloads + loads the cross-encoder model
(`cross-encoder/ms-marco-MiniLM-L-6-v2`); it is then held in memory. Re-run the
harness or the A/B runner to feel the difference.

---

## 6. Launch the watcher GUI

```bash
python -m web.server
# AIAR watcher serving on http://127.0.0.1:8088
```

Open <http://127.0.0.1:8088>. Four pages:

- **Simulate** — run a prompt, see the answer + verdict, mark it for evaluation.
- **Activity** — every LLM call the harness logged; mark any one for evaluation.
- **Evaluation queue** — score marked answers 1–10 and reground, or clear the pending queue.
- **Settings** — switch the active Qwen model (from your installed Ollama models),
  switch the active RAG instance (or pick **No RAG**), and edit the harness
  system prompt — all live, no restart.

(Host/port via `AIAR_WEB_HOST` / `AIAR_WEB_PORT`.)

**Headless / no-browser note:** the core loop is already CLI-first
(`aiar.rag.ingest`, `aiar.harness`, `aiar.eval.runner`). There is no separate
first-class CLI for Activity / queue / Settings today, but the watcher exposes
those same actions over a local JSON API once `python -m web.server` is
running. That means on Ubuntu LTS or any headless box you can still SSH-forward
the port and use `curl` instead of a browser.

---

## 7. Simulate a prompt, see the response, mark + evaluate it

On the **Simulate** page:

1. Type a prompt, e.g. *"Is live chat support available on weekends?"*
2. **Uncheck "Use RAG"** for this first run, so the answerer is blind and likely
   to get it wrong — this gives you something to correct.
3. Click **Run prompt**. You'll see the answer, the judge badge (good / partial
   / bad), and the latency + `call_id`.
4. Click **Mark for evaluation**.

Now open the **Evaluation queue** page. The marked answer is listed with its
prompt and response.

1. Set the **Score** to a low value (e.g. **3**). A **Correction** box appears
   (required for scores at/below `AIAR_REASON_THRESHOLD`, default 7).
2. Write what the answer SHOULD have been, e.g.
   *"Live chat is weekdays only, Mon–Fri 9–6 ET. There is no weekend live chat."*

---

## 8. Click Reground to feed evaluated pairs back into grounding

Click **Submit + Reground**. This:

- appends your verdict to `~/.aiar/verdicts.jsonl`, and
- records the correction into the **grounding store**
  (`~/.aiar/grounding/<hash>.json`) keyed by the prompt's normalized signature.

The card flashes green and the item leaves the queue.

> Under the hood this is the same `aiar.grounding.store.record(...)` call the
> harness reads back via `aiar.grounding.reinject.grounding_block(...)`.

---

## 9. Verify the regrounded answer improved

Back on the **Simulate** page, ask the **same prompt again** — this time with
**"Reground"** checked (and optionally "Use RAG" too):

1. Type *"Is live chat support available on weekends?"*
2. Check **Reground**.
3. Click **Run prompt**.

The answer now incorporates your correction (you'll see a green **"Reground:
applied"** badge), and the judge verdict should move up (e.g. `bad → good`).

You can confirm the same thing from the CLI:

```bash
# Without reground (baseline):
python -m aiar.harness "Is live chat support available on weekends?" --no-rag

# With reground applied (after step 8):
python -m aiar.harness "Is live chat support available on weekends?" --no-rag --reground
```

The second answer reflects the grounded correction. That's the full loop:
**ingest → retrieve → answer → judge → evaluate → reground → verify**.

---

## Appendix: optional HTTP harness service

If you'd rather drive the harness over HTTP (instead of the in-process CLI/GUI):

```bash
pip install -r requirements-service.txt
uvicorn aiar.harness.service:app --port 8765

# Answer + judge:
curl -s -X POST localhost:8765/eval/prompt \
  -H 'content-type: application/json' \
  -d '{"prompt":"How long is the refund window?"}'

# Same prompt, blind (A/B):  POST /eval/prompt?rag=false
# Optional per-request corpus: include "instance":"mydocs" in either payload.
# Record a correction:       POST /reground {"prompt":"...","score":3,"correction":"...","instance":"mydocs"}
```

## Appendix: headless watcher API

If you want the watcher's Activity / queue / Settings features without using a
browser, start it as usual:

```bash
python -m web.server
```

Then drive the same features over its local JSON API:

```bash
# Simulate a prompt through the watcher:
curl -s -X POST localhost:8088/api/simulate \
  -H 'content-type: application/json' \
  -d '{"prompt":"How many days do I have to request a refund?","rag":true}'

# Activity and queue:
curl -s localhost:8088/api/activity
curl -s "localhost:8088/api/activity/detail?call_id=<call-id>"
curl -s -X POST localhost:8088/api/activity/evaluate \
  -H 'content-type: application/json' \
  -d '{"call_id":"<call-id>"}'
curl -s localhost:8088/api/evaluation/queue
curl -s -X POST localhost:8088/api/evaluation/verdict \
  -H 'content-type: application/json' \
  -d '{"call_id":"<call-id>","score":3,"correction":"The refund window is 30 days."}'
curl -s -X POST localhost:8088/api/evaluation/clear \
  -H 'content-type: application/json' \
  -d '{}'

# Live Settings actions:
curl -s localhost:8088/api/models
curl -s -X POST localhost:8088/api/models/active \
  -H 'content-type: application/json' \
  -d '{"model":"qwen2.5:7b"}'
curl -s localhost:8088/api/rag/instances
curl -s -X POST localhost:8088/api/rag/active \
  -H 'content-type: application/json' \
  -d '{"name":"default"}'
curl -s localhost:8088/api/system-prompt
curl -s -X POST localhost:8088/api/system-prompt \
  -H 'content-type: application/json' \
  -d '{"text":"You are a precise assistant."}'
```

## Appendix: reset everything

```bash
rm -rf ~/.aiar        # vector store + grounding + logs + eval queue + verdicts
# Windows (PowerShell):  Remove-Item -Recurse -Force $HOME\.aiar
```

## License

AIAR is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).
