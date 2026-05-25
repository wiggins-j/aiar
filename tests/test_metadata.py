"""Tests for the metadata-schema framework (aiar/rag/metadata.py)."""
from __future__ import annotations

import json

from aiar.rag import metadata


SAMPLE = """---
source_domain: nhtsa.gov
source_url: https://example.gov/x
document_type: recall
authoritative: true
source_tier: 1
model: [Model 3, Model Y]
model_year_range: "2023"
loinc:
---

# Body text
This is the document body.
"""


def test_parse_front_matter_types():
    meta, body = metadata.parse_front_matter(SAMPLE)
    assert meta["source_domain"] == "nhtsa.gov"
    assert meta["authoritative"] is True            # bool
    assert meta["source_tier"] == 1                 # int
    assert meta["model"] == ["Model 3", "Model Y"]  # list
    assert meta["model_year_range"] == "2023"       # quoted -> stays string
    assert meta["loinc"] is None                    # empty -> null
    assert "Body text" in body


def test_parse_front_matter_absent():
    meta, body = metadata.parse_front_matter("# no front matter\ntext")
    assert meta == {}
    assert body.startswith("# no front matter")


def _schema():
    return {
        "name": "t",
        "fields": {
            "source_url": {"required": True, "type": "str"},
            "source_tier": {"required": True, "type": "int"},
            "document_type": {"required": True, "enum": ["recall", "spec"]},
            "region": {"required": False, "type": "str"},
        },
    }


def test_validate_metadata_valid():
    meta = {"source_url": "u", "source_tier": 1, "document_type": "recall"}
    assert metadata.validate_metadata(meta, _schema()) == []


def test_validate_metadata_required_missing():
    meta = {"source_url": "u", "document_type": "recall"}  # no source_tier
    issues = metadata.validate_metadata(meta, _schema())
    assert any("source_tier" in i and "required" in i for i in issues)


def test_validate_metadata_enum_and_type():
    meta = {"source_url": "u", "source_tier": "high", "document_type": "blog"}
    issues = metadata.validate_metadata(meta, _schema())
    assert any("source_tier" in i and "int" in i for i in issues)         # type
    assert any("document_type" in i and "allowed" in i for i in issues)   # enum


def test_validate_metadata_list_enum():
    schema = {"fields": {"tags": {"enum": ["a", "b"]}}}
    assert metadata.validate_metadata({"tags": ["a", "b"]}, schema) == []
    bad = metadata.validate_metadata({"tags": ["a", "zzz"]}, schema)
    assert any("zzz" in i for i in bad)


def test_template_from_schema():
    tpl = metadata.template_from_schema(_schema())
    assert tpl.startswith("---") and tpl.rstrip().endswith("---")
    assert "source_url:" in tpl
    assert "one of: recall | spec" in tpl


def test_validate_folder_and_cli(tmp_path):
    (tmp_path / "good.md").write_text(
        "---\nsource_url: u\nsource_tier: 1\ndocument_type: recall\n---\nbody",
        encoding="utf-8")
    (tmp_path / "bad.md").write_text(
        "---\nsource_url: u\ndocument_type: blog\n---\nbody", encoding="utf-8")
    schema_path = tmp_path / "s.json"
    schema_path.write_text(json.dumps(_schema()), encoding="utf-8")

    results = metadata.validate_folder(tmp_path, metadata.load_schema(schema_path))
    assert results[str(tmp_path / "good.md")] == []
    assert results[str(tmp_path / "bad.md")]  # has issues

    # CLI: validate returns non-zero when a file fails, zero when all pass
    assert metadata.main(["validate", str(tmp_path), "--schema", str(schema_path)]) == 1
    (tmp_path / "bad.md").unlink()
    assert metadata.main(["validate", str(tmp_path), "--schema", str(schema_path)]) == 0


def test_shipped_schemas_load():
    from pathlib import Path
    d = Path(__file__).resolve().parents[1] / "examples" / "metadata-schemas"
    for name in ("generic.schema.json", "tesla.schema.json"):
        s = metadata.load_schema(d / name)
        assert s.get("name") and "fields" in s


def test_cli_scaffold(tmp_path):
    out = tmp_path / "my.schema.json"
    assert metadata.main(["scaffold", "mine", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["name"] == "mine" and "fields" in data
