"""A/B eval runner: run a case file RAG-on vs RAG-off and report the deltas.

A case file is JSON:

    {
      "case_set_id": "my-faq",
      "cases": [
        {
          "id": "refund-window",
          "prompt": "How long is the refund window?",
          "rubric": [
            {"id": "mentions-30", "weight": 1, "match_any": ["30 day", "thirty day"]},
            {"id": "no-hallucination", "weight": 1, "forbid_any": ["lifetime", "no refunds"]}
          ]
        }
      ]
    }

For each case the runner answers the prompt twice through the in-process
harness — once with retrieval ON, once OFF — scores both with the deterministic
rubric (and optionally the LLM judge), and prints the per-variant aggregate plus
the RAG-on minus RAG-off delta. Positive rubric delta = RAG helped.

This is intentionally model-free at the orchestration layer: it imports the
harness, which imports the Ollama client. Pass ``judge=True`` to also collect an
LLM-as-judge verdict per answer.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scorer import ScoreResult, score_answer

logger = logging.getLogger(__name__)


@dataclass
class VariantResult:
    rubric_score: int = 0
    rubric_max: int = 0
    cases_passed: int = 0
    cases_total: int = 0
    latency_ms_total: int = 0

    @property
    def rubric_pct(self) -> float:
        return round(self.rubric_score / self.rubric_max, 4) if self.rubric_max else 0.0


@dataclass
class CaseOutcome:
    case_id: str
    prompt: str
    rag_on: Dict[str, Any] = field(default_factory=dict)
    rag_off: Dict[str, Any] = field(default_factory=dict)


def _answer(prompt: str, *, rag: bool, judge: bool) -> Dict[str, Any]:
    """Answer one prompt through the harness. Lazy import keeps eval light."""
    from aiar.harness import answer_prompt
    result = answer_prompt(prompt, rag=rag, judge=judge)
    return result


def run_case_file(path: Path, *, judge: bool = False,
                  progress=None) -> Dict[str, Any]:
    """Run every case RAG-on and RAG-off; return an aggregate report dict."""
    data = json.loads(path.read_text(encoding="utf-8"))
    case_set_id = str(data.get("case_set_id") or path.stem)
    cases = data.get("cases") or []

    on = VariantResult()
    off = VariantResult()
    outcomes: List[CaseOutcome] = []

    for case in cases:
        cid = str(case.get("id") or "case")
        prompt = str(case.get("prompt") or "")
        rubric = case.get("rubric") or []
        if progress:
            progress(f"running case {cid} ...")

        on_ans = _answer(prompt, rag=True, judge=judge)
        off_ans = _answer(prompt, rag=False, judge=judge)
        on_score: ScoreResult = score_answer(on_ans.get("answer", ""), rubric)
        off_score: ScoreResult = score_answer(off_ans.get("answer", ""), rubric)

        for variant, ans, sc in ((on, on_ans, on_score), (off, off_ans, off_score)):
            variant.rubric_score += sc.score
            variant.rubric_max += sc.max_score
            variant.cases_total += 1
            variant.cases_passed += 1 if sc.passed else 0
            variant.latency_ms_total += int(ans.get("latency_ms") or 0)

        outcomes.append(CaseOutcome(
            case_id=cid, prompt=prompt,
            rag_on={"answer": on_ans.get("answer"), "score": on_score.to_dict(),
                    "verdict": on_ans.get("verdict")},
            rag_off={"answer": off_ans.get("answer"), "score": off_score.to_dict(),
                     "verdict": off_ans.get("verdict")},
        ))
        if progress:
            progress(f"  -> {cid}: rag_on {on_score.score}/{on_score.max_score}  "
                     f"rag_off {off_score.score}/{off_score.max_score}")

    return {
        "case_set_id": case_set_id,
        "cases_total": len(cases),
        "variants": {
            "rag_on": {"rubric_score": on.rubric_score, "rubric_max": on.rubric_max,
                       "rubric_pct": on.rubric_pct, "cases_passed": on.cases_passed},
            "rag_off": {"rubric_score": off.rubric_score, "rubric_max": off.rubric_max,
                        "rubric_pct": off.rubric_pct, "cases_passed": off.cases_passed},
        },
        "rubric_score_delta": on.rubric_score - off.rubric_score,
        "rubric_pct_delta": round(on.rubric_pct - off.rubric_pct, 4),
        "cases": [{"case_id": o.case_id, "prompt": o.prompt,
                   "rag_on": o.rag_on, "rag_off": o.rag_off} for o in outcomes],
    }


def render(report: Dict[str, Any]) -> str:
    v = report["variants"]
    lines = [
        f"Case set: {report['case_set_id']} ({report['cases_total']} cases)",
        f"  RAG ON : {v['rag_on']['rubric_score']}/{v['rag_on']['rubric_max']} "
        f"({v['rag_on']['rubric_pct']:.0%})  passed {v['rag_on']['cases_passed']}",
        f"  RAG OFF: {v['rag_off']['rubric_score']}/{v['rag_off']['rubric_max']} "
        f"({v['rag_off']['rubric_pct']:.0%})  passed {v['rag_off']['cases_passed']}",
        f"  DELTA  : rubric {report['rubric_score_delta']:+d}  "
        f"pct {report['rubric_pct_delta']:+.1%}  (positive = RAG helped)",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aiar.eval.runner",
        description="A/B eval: run a case file RAG-on vs RAG-off.")
    parser.add_argument("case_file", help="Path to a JSON case file")
    parser.add_argument("--judge", action="store_true",
                        help="Also collect an LLM-as-judge verdict per answer")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from aiar.rag import store
    store.init()

    path = Path(args.case_file).expanduser()
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    report = run_case_file(path, judge=args.judge, progress=lambda m: print(m))
    print()
    print(render(report))
    if args.json:
        print()
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
