<p align="center">
  <img src="web/static/aiar-logo.png" alt="AIAR logo" width="280" />
</p>

## ⚡ Quick start: hand this to your AI

New here? Don't read the docs — **copy the block below and paste it into any
capable coding AI agent** (Claude, ChatGPT/Codex, Gemini, Cursor, Copilot Chat,
etc.). It will set up the entire AIAR project for you end-to-end, asking you a
few simple questions (each with a recommended default you can just accept).

````text
You are setting up AIAR ("Artificial Intelligence and RAG") for me on my machine — a small,
open-source, model-agnostic framework for building AND EVALUATING a
retrieval-augmented (RAG) assistant over MY OWN documents, using ANY Qwen model
served locally by Ollama. The full loop is: ingest my docs → retrieve →
answer with Qwen (grounded) → judge the answer 1–10 with a reason → let me
correct it → "reground" so the next answer is fixed. License is Apache-2.0; do
not change the project's code or behavior — only set it up and run it.

Drive the whole setup for me. The authoritative steps live in PLAYBOOK.md in the
repo — follow it. Work step by step, run the real commands, and after EACH step
confirm it actually succeeded (show the relevant output) before moving on. Be
defensive: never assume — check prerequisites first and stop with a clear
explanation if something is missing.

=== STEP -1: Verify or bootstrap the repo BEFORE asking Step 0 questions ===
First confirm you are in a real AIAR checkout, not just an empty directory or a
bare/partial Git repo. Check the working directory and verify that BOTH
`PLAYBOOK.md` and `config.example` are present there before doing anything else.

If they are missing:
- Show me the exact path you checked and the directory listing.
- Check whether this is just the wrong local path by looking for another nearby
  populated AIAR checkout.
- If you find a populated checkout, tell me the exact correct path and switch to it.
- If the directory only contains `.git`, has no commits, or is otherwise not
  populated, treat it as a broken/incomplete repo bootstrap and FIX IT YOURSELF:
  clone the canonical AIAR repo into that path, using HTTPS by default:
  `https://github.com/wiggins-j/aiar.git`
- If the target directory is unusable for cloning because it already contains an
  empty `.git` directory, remove that broken directory and re-clone cleanly into
  the same path.
- Do NOT stop and ask me for the repo URL unless cloning the canonical repo
  actually fails.
- Only after the repo contents are present and BOTH `PLAYBOOK.md` and
  `config.example` exist should you continue to Step 0.

=== STEP 0: Ask me these configuration questions FIRST ===
Ask me all of these up front. For each, tell me the recommended default and let
me just say "defaults" to accept them all. Then use my answers (or the defaults)
throughout. Set each as an environment variable (see config.example for the
exact names) — AIAR is entirely environment-driven.

1. What hardware will run the model, and which Qwen should I use? — FIRST either
   tell me your specs (total RAM; GPU + VRAM if any; free disk), OR let me
   AUTO-DETECT them: say "scan local" and I'll run the OS-appropriate probe, or
   "scan remote <user@host>" and I'll probe over SSH:
     • Linux:   `free -h` ; `nvidia-smi --query-gpu=memory.total --format=csv` ; `df -h ~`
     • macOS:   `sysctl hw.memsize` ; `system_profiler SPDisplaysDataType | grep VRAM` ; `df -h ~`
                (Apple Silicon shares ONE "unified memory" pool = your RAM)
     • Windows: `systeminfo | findstr /C:"Total Physical Memory"` ; `nvidia-smi` ; `Get-PSDrive C`
   Then I'll recommend the LARGEST Qwen that fits, from the table below, and set
   OLLAMA_MODEL to a tag that actually exists for your Ollama. ANY Qwen works.

   Qwen size → hardware (rough, Q4_K_M quant — VERIFY exact + newest at the URLs below):
     ~1.5B   `qwen2.5:1.5b`              ~2 GB RAM  / ~2 GB VRAM   tiny boxes, smoke tests
     ~3B     `qwen2.5:3b`               ~4 GB RAM  / ~4 GB VRAM   laptops, fast iteration
     ~7–8B   `qwen2.5:7b`, `qwen3:8b`   ~8 GB RAM  / ~6 GB VRAM   balanced default ★
     ~14B    `qwen2.5:14b`, `qwen3:14b` ~16 GB RAM / ~12 GB VRAM  stronger reasoning
     ~32B    `qwen2.5:32b`, `qwen3:32b` ~32 GB RAM / ~24 GB VRAM  workstation / server
     ~72B    `qwen2.5:72b`              ~48 GB+ RAM / 2×24 GB     multi-GPU server
   No GPU? CPU-only runs at every size — just slower; pick one tier below your RAM row.

   ⚠️ ALWAYS check the canonical sources for up-to-date models + exact specs — the
   table above is a SNAPSHOT and will age:
     • Qwen model cards & specs:  https://huggingface.co/Qwen
     • Qwen 3.5 collection:       https://huggingface.co/collections/Qwen/qwen35
     • Qwen 3.6 collection:       https://huggingface.co/collections/Qwen/qwen36
     • Ollama-pullable tags:      https://ollama.com/library/qwen
   The exact tag must exist for YOUR Ollama version — if any tag referenced in this
   repo isn't pullable, substitute the nearest real one from those URLs.
   → sets OLLAMA_MODEL
2. Where is my document corpus (a folder of .txt/.md/.markdown/.rst/.json)? —
   (recommended: start with the bundled `./examples/docs` to prove the loop
   works, then re-run ingest pointed at my real folder.)
3. RAG corpus / collection name? — (recommended: `aiar`.) → sets AIAR_CORPUS
4. Embedding model? — (recommended: `all-MiniLM-L6-v2`, the default.)
   → sets AIAR_EMBEDDING_MODEL
5. How many chunks to inject into each prompt (top-k)? — (recommended: `3`.)
   → sets RAG_TOP_K
6. Enable HYBRID retrieval (dense vectors + BM25)? — (recommended: ON, great for
   exact terms.) → sets RAG_HYBRID_ENABLED=1
7. Enable cross-encoder RERANKER? — (recommended: ON for relevance; first use
   downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` and a wide first pass of
   RAG_FETCH_K=20.) → sets RAG_RERANK_ENABLED=1, RAG_FETCH_K=20
8. Query rewrite / HYDE mode (off | rewrite | hyde)? — (recommended: `hyde`,
   closes the vocabulary gap.) → sets RAG_QUERY_REWRITE_MODE
9. Enable GROUNDING reinjection (auto-apply my past corrections to every
   answer)? — (recommended: ON so corrections stick.)
   → sets GROUNDING_REINJECTION_ENABLED=1
   (Note: the project ships ALL of flags 6–9 defaulting OFF so the bare path is
   plain vector search; we turn them on here for quality. That's expected.)
10. GUI host and port? — (recommended: host `127.0.0.1`, port `8088`, the
    config.example defaults.) → sets AIAR_WEB_HOST, AIAR_WEB_PORT
11. Correction-required score threshold (a score at/below this needs a written
    correction)? — (recommended: `7`.) → sets AIAR_REASON_THRESHOLD
12. Do I also want the optional HTTP harness service (FastAPI)? — (recommended:
    NO for now; the CLI + GUI need none of it. If yes, also
    `pip install -r requirements-service.txt` and run
    `uvicorn aiar.harness.service:app --port 8765`.)
Also note: the store persists at `~/.aiar` (AIAR_DB_PATH / AIAR_BASE_DIR); to
reset everything later, `rm -rf ~/.aiar` (Windows PowerShell:
`Remove-Item -Recurse -Force $HOME\.aiar`).

=== STEP 1: Check prerequisites (do NOT assume) ===
- OS: AIAR is OS-agnostic — Linux (incl. Ubuntu LTS servers), macOS, and Windows
  all work (pure Python + pathlib; no shell scripts, no platform branches).
  Detect mine and use the matching commands throughout (venv activation, paths,
  reset). On a HEADLESS Ubuntu server there's no browser for the GUI — either
  SSH-forward the GUI port (`ssh -L 8088:127.0.0.1:8088 me@server`) or set
  `AIAR_WEB_HOST=0.0.0.0` behind a firewall/tunnel; the CLI needs no GUI.
- Python is 3.10–3.14. On newer active versions, the RAG stack may need to
  build `chroma-hnswlib` from source on some platforms if a wheel is
  unavailable there. Run `python3 --version` (Windows: `py --version`) and
  confirm.
- `ollama` is installed and the server is reachable. If `ollama` is missing,
  tell me how to install it for my OS and have me confirm:
    • macOS:   `brew install ollama` or the .dmg from https://ollama.com/download
    • Linux:   `curl -fsSL https://ollama.com/install.sh | sh`
    • Windows: the installer from https://ollama.com/download (runs as a service)
  Make sure the server is running (`ollama serve &`, the menubar/tray app, or the
  Windows service); `ollama list` should respond.
- Confirm I have enough free disk/RAM/VRAM for the model chosen in STEP 0 (use
  the size→hardware table). If it won't fit, recommend the next tier down.

=== STEP 2: Pull the Qwen model ===
`ollama pull <my model>` (model-agnostic — whatever I chose). Verify with
`ollama list`.

=== STEP 3: Install AIAR ===
From the repo root: create and activate a venv, then install core + RAG stack:
  python -m venv .venv
  # activate — Linux/macOS:  source .venv/bin/activate
  #            Windows (PowerShell):  .venv\Scripts\Activate.ps1
  #            Windows (cmd):         .venv\Scripts\activate.bat
  pip install -r requirements.txt -r requirements-rag.txt
(`requirements.txt` is the core; `requirements-rag.txt` adds the vector store,
embeddings, BM25, and reranker — install both.) Then export OLLAMA_MODEL and the
other config vars from STEP 0 (you can `source config.example` and edit, or set
them directly). Confirm `python -c "import aiar"` works.

=== STEP 4: Ingest my documents into the RAG ===
AIAR ingests a folder of documents. Two ways to fill that folder:
  (a) MANUAL — I already have a folder of .txt/.md/.json files.
  (b) AI-DRIVEN — an AI builds the corpus from a "Collection Brief" (see the
      README section "Building your RAG corpus: two ways" +
      examples/corpus-briefs/). If I want this, offer to either copy the Tesla
      example brief or run the builder prompt to interview me, save my brief to
      briefs/<name>-collection-brief.md, collect docs into corpus/<name>/, then
      ingest that folder.
Preview first (no writes): `python -m aiar.rag.ingest <docs folder> --dry-run`
Then ingest for real: `python -m aiar.rag.ingest <docs folder> --instance <name>`
(Omit --instance to use the active/`default` instance; `--category <name>` adds a
metadata tag.) Embeddings persist to `~/.aiar/knowledge`. Confirm chunks written.

=== STEP 5: Run the harness ===
`python -m aiar.harness "How many days do I have to request a refund?"`
(use a question relevant to MY docs). Show me the answer, the judge verdict
(rating + reason + confidence), and the call_id. Useful flags to mention:
`--no-rag` (blind A/B baseline), `--think` (show reasoning), `--json` (full
result), `--reground` (apply prior corrections), `--top-k N`.
Then show retrieval lift with the A/B runner:
`python -m aiar.eval.runner ./examples/cases.json` (reports RAG-on vs RAG-off
delta; positive = RAG helped).

=== STEP 6: Confirm the quality flags are on ===
Confirm the env vars from STEP 0 (RAG_HYBRID_ENABLED, RAG_RERANK_ENABLED +
RAG_FETCH_K, RAG_QUERY_REWRITE_MODE, GROUNDING_REINJECTION_ENABLED) are set,
then re-run the harness so the reranker model loads and the difference is felt.

=== STEP 7: Launch the watcher GUI ===
`python -m web.server` → open http://127.0.0.1:8088 (or my host/port). It has
four pages: Simulate (run a prompt, see answer + verdict, mark it), Activity
(every logged LLM call, mark any one), Evaluation queue (score 1–10 + correct),
and Settings (switch the active Qwen model, switch the active RAG instance —
including a first-class "No RAG" option — and edit the harness system prompt).
Tell me it's serving and what each page does.

=== STEP 8: Verify the full end-to-end loop ===
Walk me through it so I SEE it work:
1. On Simulate, run a prompt with "Use RAG" UNCHECKED so the blind answer is
   likely wrong (e.g. "Is live chat support available on weekends?"). Mark it
   for evaluation.
2. On the Evaluation queue, set a LOW score (e.g. 3) — a Correction box appears
   (required at/below the threshold) — write what the answer SHOULD say, then
   click "Submit + Reground". This writes to ~/.aiar/verdicts.jsonl and the
   grounding store (~/.aiar/grounding/<hash>.json).
3. Back on Simulate, ask the SAME prompt again with "Reground" checked: the
   answer should incorporate my correction (green "Reground: applied" badge) and
   the verdict should move up (e.g. bad → good).
You can also verify from the CLI: run the prompt with `--no-rag`, then again
with `--no-rag --reground`, and show me the second answer reflects the
correction. That's the full loop: ingest → retrieve → answer → judge →
evaluate → reground → verify.

=== FINISH: Summarize ===
When everything passes, give me a short summary of how to use AIAR day to day:
the ingest command for new docs, how to run the harness/A/B runner, how to open
the GUI, how the reground loop works, which env vars control behavior (point me
at config.example), and that `rm -rf ~/.aiar` resets all state.
````

---

# AIAR — Artificial Intelligence and RAG

A small, open-source, **model-agnostic** framework for building and *evaluating*
a retrieval-augmented (RAG) assistant on **your own documents**, using **any
Qwen model via [Ollama](https://ollama.com)**.

AIAR gives you the full loop in one repo:

1. **Ingest** a folder of YOUR documents (`.txt` / `.md` / `.json`) into a
   vector store.
2. **Retrieve** with a real pipeline — dense vector search, optional **hybrid**
   (vector + BM25), optional **cross-encoder reranking**, optional query
   rewrite / HyDE.
3. **Answer** a prompt with your Qwen model, grounded in the retrieved context.
4. **Judge** the answer with an LLM-as-judge (rating + reason + confidence).
5. **Simulate / evaluate / reground** through a web GUI: run a prompt, see the
   response, mark it, score it 1–10 with a correction, then click **Submit +
   Reground** to feed evaluated pairs back into a **grounding store** so the next
   answer is corrected.

Nothing here is tied to any application domain — point it at your documents and
go.

## Architecture

The full loop in one repo — ingest, retrieve, ground, answer, judge, reground:

```mermaid
flowchart LR
    subgraph INGEST["Ingest (offline)"]
        DOCS["Your documents<br/>.txt · .md · .json"] --> ING["aiar.rag.ingest<br/>chunk + embed"]
    end
    ING --> STORE[("ChromaDB store<br/>N named RAG instances")]

    Q(["User prompt"]) --> RET["Retriever<br/>vector + BM25 (hybrid)<br/>cross-encoder rerank<br/>query-rewrite / HyDE"]
    STORE --> RET
    GND[("Grounding store<br/>past corrections")] -- reground --> CTX["Context<br/>(chunks + corrections)"]
    RET --> CTX
    CTX --> HAR["Harness: answer_prompt<br/>answer with any Qwen via Ollama"]
    HAR --> ANS(["Answer"])
    ANS --> JUDGE["LLM-as-judge<br/>rating + reason + confidence"]
    JUDGE --> GUI["Watcher GUI / CLI<br/>Simulate · Activity · Evaluation"]
    GUI -- "score 1–10 + correction" --> GND

    SET[["Settings<br/>active model · RAG instance · system prompt"]] -.-> HAR
    SET -.-> RET
    OBS["Observability<br/>JSONL call log"] -.-> GUI
    HAR -.-> OBS
```

- **Ingest** is offline: your docs → chunks → embeddings in ChromaDB (one or many
  named **RAG instances**; "No RAG" is also selectable).
- **Answer time** retrieves (optionally hybrid + reranked + rewritten), prepends any
  prior **grounding** corrections, answers with your Qwen model, then **judges** it.
- The **GUI/CLI** closes the loop: a low score + correction is written back to the
  grounding store, so the next answer to that prompt is fixed (**reground**).
- **Settings** swaps the active model, RAG instance, and harness system prompt live.

## Why "Artificial Intelligence and RAG"?

Most RAG demos stop at "retrieve and answer." AIAR is about the part that
matters in practice: *measuring whether retrieval and grounding actually made
the answer better*, and *closing the loop* when they didn't.

## Components

| Package | What it is |
|---|---|
| `aiar/rag` | `ingest` (folder → chunks), `store` (ChromaDB), `retriever` (`get_context`), `lexical` (BM25), `fusion` (RRF), `reranker` (cross-encoder), `query_rewrite` (HyDE) |
| `aiar/llm` | `call_ollama` — the only thing that talks to Ollama; model-agnostic |
| `aiar/eval` | `judge` (LLM-as-judge), `scorer` (deterministic rubric), `runner` (RAG-on vs RAG-off A/B) |
| `aiar/grounding` | `store` (corrections keyed by prompt signature), `reinject` (render corrections into the next prompt) |
| `aiar/harness` | `pipeline.answer_prompt` (prompt → retrieve → reground → answer → judge), a CLI, and an optional FastAPI `service` |
| `aiar/observability` | JSONL logging of every LLM call (the GUI tails this) |
| `web/` | the stdlib-only watcher GUI: Simulate / Activity / Evaluation / Settings pages |

## Quickstart

Use the Python launcher that exists on your machine: on many macOS/Linux setups
that means `python3`; on Windows it is often `py`. The examples below use
`python` generically.

```bash
# 1. Pull any Qwen model with Ollama. Pick a tag sized to your hardware; verify
#    it's pullable at https://ollama.com/library/qwen
#    (specs / collections: https://huggingface.co/Qwen ,
#     https://huggingface.co/collections/Qwen/qwen35 ,
#     https://huggingface.co/collections/Qwen/qwen36)
ollama pull qwen2.5:7b          # or qwen2.5:3b (laptops), qwen3:14b (stronger), ...

# 2. Install AIAR + the RAG stack
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-rag.txt

# 3. Tell AIAR which model to use (the tag you pulled above)
export OLLAMA_MODEL="qwen2.5:7b"                     # Windows: $env:OLLAMA_MODEL="qwen2.5:7b"

# 4. Ingest YOUR documents
python -m aiar.rag.ingest ./examples/docs

# 5. Ask a question
python -m aiar.harness "How many days do I have to request a refund?"

# 6. Launch the GUI (Simulate → mark → evaluate → reground)
python -m web.server      # http://127.0.0.1:8088
```

See **[PLAYBOOK.md](PLAYBOOK.md)** for the complete, copy-paste, end-to-end
walkthrough — including turning on the reranker + grounding flags and verifying
a regrounded answer improved.

## Building your RAG corpus: two ways

AIAR ingests a *folder of documents*. How that folder gets populated is up to you —
there are two supported paths:

**1. Manual — you gather the documents.** Drop `.txt` / `.md` / `.markdown` / `.rst`
/ `.json` files into a folder and ingest:

```bash
python -m aiar.rag.ingest /path/to/my/docs --instance my-corpus
```

(Omit `--instance` to ingest into the active instance, which defaults to
`default`. A new named instance is created on first ingest.)

**2. AI-driven — an AI builds the corpus from a brief.** Write a one-file
**Collection Brief** that tells an AI agent (one with web + file tools) exactly
which sources to use, what's in/out of scope, how to split + tag files, and the
safety rules to respect. Hand the brief to the agent; it fetches and normalizes
documents into a folder; then you ingest that folder with AIAR as above.

- **Worked example:** [`examples/corpus-briefs/tesla-manual-expert-collection-brief.md`](examples/corpus-briefs/tesla-manual-expert-collection-brief.md)
  — a safety-first brief for a Tesla owner-manual assistant (official Tesla + NHTSA
  sources only, safety classes, metadata schema, eval questions).
- **Generate your own:** paste [`examples/corpus-briefs/collection-brief-builder-prompt.md`](examples/corpus-briefs/collection-brief-builder-prompt.md)
  into any AI — it interviews you and writes a brief like the Tesla one for your domain.

**Where to put the files:**

| File | Location | Tracked? |
|---|---|---|
| Your collection brief | `briefs/<name>-collection-brief.md` (create `briefs/`) | yours to keep / version |
| The collected documents | `corpus/<name>/` (one topic/procedure per file) | **git-ignored** (large / regenerable) |

Then ingest: `python -m aiar.rag.ingest corpus/<name> --instance <name>`.

> Keep the brief **outside** the ingested folder (`briefs/`, not `corpus/<name>/`)
> so the brief itself isn't indexed as a document. AIAR does the chunk + embed +
> index; the brief only governs what the AI collects and how it splits/tags files.
> See [`examples/corpus-briefs/README.md`](examples/corpus-briefs/README.md) for the
> full workflow.

## Configuration

Everything is environment-driven; see [`config.example`](config.example) for the
full list. The retrieval-quality flags (`RAG_HYBRID_ENABLED`,
`RAG_RERANK_ENABLED`, `RAG_QUERY_REWRITE_MODE`) and grounding
(`GROUNDING_REINJECTION_ENABLED`) all default OFF so the bare path is plain
vector retrieval — turn them on to trade a little latency for relevance.

## Requirements

- **OS-agnostic** — Linux (incl. Ubuntu LTS servers), macOS, and Windows. Pure
  Python + `pathlib`; no shell scripts or platform-specific code.
- Python 3.10–3.14
- [Ollama](https://ollama.com) running locally with any Qwen model pulled. Pick a
  model sized to your hardware — tags at <https://ollama.com/library/qwen>, specs
  at <https://huggingface.co/Qwen>, with newer family collections at
  <https://huggingface.co/collections/Qwen/qwen35> and
  <https://huggingface.co/collections/Qwen/qwen36> (always check these for the
  latest models).
- `requirements.txt` (core) + `requirements-rag.txt` (vector store + reranker).
  The RAG stack is optional: without it the store is unavailable and the harness
  answers from the bare model.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
