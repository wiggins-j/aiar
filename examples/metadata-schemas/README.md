# Metadata schemas

A **metadata schema** is a small JSON file that declares the front-matter every
record in a RAG corpus should carry — source, URL, trust tier, claim type, dates,
codes — so your knowledge base stays consistent and its **trust signals are
reliable** (official label > guideline > trial > observational > anecdote).

This is the framework half of a [collection brief](../corpus-briefs/): the brief
tells an AI collector *what to write*; the schema lets you *check it wrote it
right*. The schema engine + CLI live in
[`aiar/rag/metadata.py`](../../aiar/rag/metadata.py) and are domain-agnostic — ship
one schema per collection.

## Files here

| File | What it is |
|---|---|
| [`generic.schema.json`](generic.schema.json) | Domain-agnostic starter. Copy it and add your fields/enums. |
| [`tesla.schema.json`](tesla.schema.json) | Worked example matching the Tesla open-data brief. |

## Schema format

```json
{
  "name": "my-collection",
  "description": "what this corpus is",
  "fields": {
    "source_url":  {"required": true,  "type": "str"},
    "source_tier": {"required": true,  "type": "int"},
    "claim_type":  {"required": true,  "enum": ["guideline", "label", "trial"]},
    "region":      {"required": false, "type": "str"}
  }
}
```

Per-field keys: `required` (bool), `type` (`str|int|float|bool|list`), `enum`
(allowed values; for a list field every item must be in the enum), `description`
(free text, shown in `template`).

## CLI

```bash
# 1) Scaffold a schema to edit:
python -m aiar.rag.metadata scaffold my-collection --out my.schema.json

# 2) Print a blank front-matter block to paste into new records:
python -m aiar.rag.metadata template --schema my.schema.json

# 3) Validate a collected corpus folder against the schema:
python -m aiar.rag.metadata validate corpus/mydocs --schema my.schema.json
#    -> lists missing required fields, bad enum values, and type mismatches,
#       then "N/M files valid". Exit code is non-zero if any file fails.
```

Front-matter is the YAML block between the first two `---` lines of a file. The
parser handles the flat shapes the briefs use (`key: value`, `key: [a, b]`,
`key:`/`key: null`); quote a value to force it to stay a string (e.g.
`model_year_range: "2023"`).

## Use it in your workflow

1. Write (or scaffold) a schema for your collection and keep it next to the brief.
2. Put the brief's "Required Metadata Fields" into the schema as `fields`.
3. After the collector writes `corpus/<name>/`, run `validate` before you ingest —
   or pass `--validate <schema.json>` to `python -m aiar.rag.ingest` to validate as
   you ingest (warnings only; it won't block).
4. Fix any flagged records so the trust tiers in your KB are dependable.
