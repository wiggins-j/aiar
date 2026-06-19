"""Contract tests for grounding record/lookup (aiar.grounding.v1).

Uses a tmp ``base`` so the real ~/.aiar grounding dir is never touched.
"""
from __future__ import annotations

from aiar.contracts.grounding import GROUNDING_SCHEMA_VERSION
from aiar.eval.schemas import Verdict
from aiar.grounding import store as gstore


def _verdict(rating="bad"):
    return Verdict(rating=rating, reason="was wrong", failure_tags=["x"],
                   confidence="high")


def test_answer_and_correction_are_distinct(tmp_path):
    rec = gstore.record_grounding(
        signature="what is 2+2", verdict=_verdict(), correction="it is 4",
        answer="it is 5", prompt="what is 2+2", source_chunks=["c1", "c2"],
        instance="alpha", base=tmp_path)
    d = rec.to_dict()
    assert d["schema_version"] == GROUNDING_SCHEMA_VERSION
    assert d["answer"] == "it is 5"
    assert d["correction"] == "it is 4"
    assert d["source_chunks"] == ["c1", "c2"]
    assert d["verdict"] == "bad"
    assert d["instance"] == "alpha"
    assert d["id"]


def test_lookup_returns_records(tmp_path):
    gstore.record_grounding(signature="q1", verdict=_verdict(),
                            correction="fix", answer="bad", instance="alpha",
                            base=tmp_path)
    out = gstore.lookup_grounding(signature="q1", instance="alpha", base=tmp_path)
    assert len(out) == 1
    assert out[0].answer == "bad"
    assert out[0].correction == "fix"


def test_instance_scoping(tmp_path):
    gstore.record_grounding(signature="shared", verdict=_verdict(),
                            correction="alpha fix", instance="alpha",
                            base=tmp_path)
    # A different corpus must never see alpha's correction.
    assert gstore.lookup_grounding(signature="shared", instance="beta",
                                   base=tmp_path) == []
    assert len(gstore.lookup_grounding(signature="shared", instance="alpha",
                                       base=tmp_path)) == 1


def test_legacy_record_is_readable(tmp_path):
    # A record written through the old positional path reads back with answer=None
    # and the original text left in correction (never rewritten).
    gstore.record("legacy q", _verdict(), "legacy correction",
                  instance="alpha", base=tmp_path)
    out = gstore.lookup_grounding(signature="legacy q", instance="alpha",
                                  base=tmp_path)
    assert len(out) == 1
    assert out[0].answer is None
    assert out[0].correction == "legacy correction"
    assert out[0].verdict == "bad"
    assert out[0].id  # synthesized stable id for legacy entries
