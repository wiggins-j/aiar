# Corpus briefs

A **collection brief** is one Markdown file that tells an AI agent exactly which
documents to find, where, and how to normalize them — so the AI builds your AIAR
corpus *for* you instead of you gathering files by hand. (See the repo
README/PLAYBOOK: **"Building your RAG corpus: two ways."**)

| File | What it is |
|---|---|
| [`tesla-manual-expert-collection-brief.md`](tesla-manual-expert-collection-brief.md) | A full **worked example** — an access-respecting Tesla knowledge-base brief built on **open APIs + public/licensed data** (NHTSA, vPIC, EPA fuel-economy, NREL charging, GDELT news, SEC; plus tiered review/forum/sales sources with trust handling). Tesla's own sites block automated agents, so they're excluded — never bypass bot protection. Includes verified endpoints, safety classes, metadata schema, and eval questions. Copy it as a starting point. |
| [`collection-brief-builder-prompt.md`](collection-brief-builder-prompt.md) | A **template prompt** — paste it into any AI and it interviews you, then writes a brief like the Tesla one for *your* domain. |

## Workflow

1. **Make your brief** — copy the Tesla example and edit it, or run the builder
   prompt and answer its questions. Save it at `briefs/<name>-collection-brief.md`
   in your AIAR checkout (create the `briefs/` folder; it's yours to keep/version).
2. **Hand the brief to an AI collector** (an agent with web + file tools). It
   fetches + normalizes documents into `corpus/<name>/` (one topic/procedure per
   `.md`/`.txt`/`.json` file, with YAML front-matter). `corpus/` is git-ignored.
3. **(Optional) Validate the metadata** — declare a schema for your brief's
   "Required Metadata Fields" and check the collected files before ingesting:
   `python -m aiar.rag.metadata validate corpus/<name> --schema my.schema.json`
   (or pass `--validate my.schema.json` to `ingest`). See
   [`../metadata-schemas/`](../metadata-schemas/).
4. **Ingest into AIAR** — `python -m aiar.rag.ingest corpus/<name> --instance <name>`.
5. **Wire it up** — put the brief's "Assistant Response Format" into the system
   prompt (Settings page), load its example questions into `examples/cases.json`,
   and run `python -m aiar.eval.runner` to measure RAG lift.

> Keep the brief OUTSIDE the folder you ingest (`briefs/`, not `corpus/<name>/`) so
> the brief itself isn't indexed as a document.
