"""Metadata-schema framework for RAG corpus files.

A *collection brief* asks an AI collector to write one record per file with YAML
**front-matter** (a leading ``---`` block) describing the source — domain, URL,
authority/trust tier, claim type, dates, codes, etc. This module lets you DECLARE
that expected metadata as a small JSON **schema** and VALIDATE that a corpus folder
conforms, so a knowledge base stays consistent and its trust signals are reliable.

It is domain-agnostic: ship a schema per collection (a Tesla schema, a health
schema, your own) and validate against it. Nothing here is tied to a domain.

Schema format (JSON)::

    {
      "name": "generic",
      "description": "what this schema is for",
      "fields": {
        "source_url":  {"required": true,  "type": "str"},
        "source_tier": {"required": true,  "type": "int"},
        "claim_type":  {"required": true,  "enum": ["guideline", "label", "trial"]},
        "region":      {"required": false, "type": "str"}
      }
    }

Per-field keys: ``required`` (bool), ``type`` (str|int|float|bool|list),
``enum`` (allowed values; for list fields every item must be in the enum), and
``description`` (free text, used by ``template``).

Front-matter is the block between the first two ``---`` lines of a file. A
deliberately small parser handles the flat ``key: value``, ``key: [a, b]`` and
``key:``/``key: null`` shapes the briefs use (no nested maps; quote a value to
force it to stay a string). Validation never raises on a bad file — it reports a
list of issues.

CLI::

    python -m aiar.rag.metadata validate <folder> --schema <schema.json>
    python -m aiar.rag.metadata template --schema <schema.json>
    python -m aiar.rag.metadata scaffold <name> [--out <file>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_FRONT_MATTER_RE = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)
_INT_RE = re.compile(r"-?\d+")
_FLOAT_RE = re.compile(r"-?\d+\.\d+")
_TYPES = {"str": str, "int": int, "float": float, "bool": bool, "list": list}

_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


# --------------------------------------------------------------------------
# Front-matter parsing (minimal, dependency-free)
# --------------------------------------------------------------------------

def _coerce_scalar(raw: str):
    s = raw.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]  # quoted -> always a string
    low = s.lower()
    if low in ("null", "~", "none", ""):
        return None
    if low in ("true", "false"):
        return low == "true"
    if _FLOAT_RE.fullmatch(s):
        return float(s)
    if _INT_RE.fullmatch(s):
        return int(s)
    return s


def _coerce(raw: str):
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(part) for part in inner.split(",")]
    return _coerce_scalar(s)


def parse_front_matter(text: str) -> Tuple[Dict[str, object], str]:
    """Return ``(metadata, body)``. ``metadata`` is ``{}`` when there is no
    leading ``---`` front-matter block. Only top-level (unindented) keys are read.
    """
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta: Dict[str, object] = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] in (" ", "\t"):  # nested / list-continuation — skipped
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        meta[key.strip()] = _coerce(val)
    return meta, text[m.end():]


# --------------------------------------------------------------------------
# Schema + validation
# --------------------------------------------------------------------------

def load_schema(path: "str | Path") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_metadata(meta: Dict[str, object], schema: dict) -> List[str]:
    """Return a list of human-readable issues (empty == valid)."""
    issues: List[str] = []
    for name, spec in (schema.get("fields") or {}).items():
        if not isinstance(spec, dict):
            continue
        present = name in meta and meta[name] is not None and meta[name] != []
        if spec.get("required") and not present:
            issues.append(f"missing required field '{name}'")
            continue
        if not present:
            continue
        val = meta[name]
        t = spec.get("type")
        if t in _TYPES and not isinstance(val, _TYPES[t]):
            # tolerate int where float is declared
            if not (t == "float" and isinstance(val, int)):
                issues.append(
                    f"field '{name}' should be {t}, got {type(val).__name__}")
        enum = spec.get("enum")
        if enum:
            values = val if isinstance(val, list) else [val]
            bad = [v for v in values if v not in enum]
            if bad:
                issues.append(f"field '{name}' has value(s) not allowed: {bad}")
    return issues


def validate_file(path: "str | Path", schema: dict) -> List[str]:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"could not read: {exc}"]
    meta, _ = parse_front_matter(text)
    if not meta:
        return ["no YAML front-matter found"]
    return validate_metadata(meta, schema)


def validate_folder(folder: "str | Path", schema: dict) -> Dict[str, List[str]]:
    """Validate every text document under ``folder``. Returns ``{path: issues}``."""
    out: Dict[str, List[str]] = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES:
            out[str(p)] = validate_file(p, schema)
    return out


def template_from_schema(schema: dict) -> str:
    """Render a blank front-matter block authors can paste into a new record."""
    lines = ["---"]
    for name, spec in (schema.get("fields") or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        hint = ""
        if spec.get("enum"):
            hint = "  # one of: " + " | ".join(str(v) for v in spec["enum"])
        elif spec.get("description"):
            hint = f"  # {spec['description']}"
        elif spec.get("type"):
            hint = f"  # {spec['type']}"
        lines.append(f"{name}:{hint}")
    lines.append("---")
    return "\n".join(lines)


STARTER_SCHEMA = {
    "name": "my-collection",
    "description": "Describe what this corpus is and how its records are tagged.",
    "fields": {
        "source_domain": {"required": True, "type": "str"},
        "source_url": {"required": True, "type": "str"},
        "retrieved_at": {"required": True, "type": "str",
                          "description": "ISO date the record was collected"},
        "document_title": {"required": True, "type": "str"},
        "document_type": {"required": True, "type": "str"},
        "authoritative": {"required": True, "type": "bool",
                           "description": "true = official/regulator; false = anecdote/estimate"},
        "source_tier": {"required": False, "type": "int",
                         "description": "trust tier (1 = highest authority)"},
        "content_hash": {"required": False, "type": "str"},
    },
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cmd_validate(args) -> int:
    schema = load_schema(args.schema)
    results = validate_folder(args.folder, schema)
    if not results:
        print("No text documents found.", file=sys.stderr)
        return 1
    bad = 0
    for path, issues in results.items():
        if issues:
            bad += 1
            print(f"FAIL {path}")
            for i in issues:
                print(f"     - {i}")
        elif args.verbose:
            print(f"ok   {path}")
    total = len(results)
    print(f"\n{total - bad}/{total} files valid against schema "
          f"{schema.get('name', '?')!r}.")
    return 0 if bad == 0 else 1


def _cmd_template(args) -> int:
    print(template_from_schema(load_schema(args.schema)))
    return 0


def _cmd_scaffold(args) -> int:
    schema = dict(STARTER_SCHEMA)
    schema["name"] = args.name
    out = json.dumps(schema, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
        print(f"Wrote starter schema to {args.out}")
    else:
        print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aiar.rag.metadata",
        description="Declare and validate metadata schemas for RAG corpus files.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate a corpus folder's front-matter")
    v.add_argument("folder")
    v.add_argument("--schema", required=True, help="path to a schema .json")
    v.add_argument("--verbose", action="store_true")
    v.set_defaults(func=_cmd_validate)

    t = sub.add_parser("template", help="print a blank front-matter block")
    t.add_argument("--schema", required=True)
    t.set_defaults(func=_cmd_template)

    s = sub.add_parser("scaffold", help="write a starter schema you can edit")
    s.add_argument("name")
    s.add_argument("--out", default=None, help="file to write (default: stdout)")
    s.set_defaults(func=_cmd_scaffold)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
