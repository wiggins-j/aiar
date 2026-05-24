"""CLI for the AIAR prompt harness.

    python -m aiar.harness "How long is the refund window?"
    python -m aiar.harness "..." --no-rag          # answer without retrieval
    python -m aiar.harness "..." --no-judge        # skip the LLM judge
    python -m aiar.harness "..." --reground        # prepend prior corrections
    python -m aiar.harness "..." --think           # show the model's reasoning
"""
from __future__ import annotations

import argparse
import json
import sys

from aiar.llm import OllamaError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aiar.harness",
        description="Answer a prompt: retrieve -> (reground) -> answer -> judge.")
    parser.add_argument("prompt", help="The prompt / question to answer")
    parser.add_argument("--no-rag", action="store_true", help="Skip retrieval (blind the answerer)")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-as-judge step")
    parser.add_argument("--think", action="store_true", help="Show the model's reasoning")
    parser.add_argument("--reground", action="store_true",
                        help="Prepend prior grounding corrections for this prompt")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = parser.parse_args(argv)

    from aiar.rag import store
    store.init()

    from aiar.harness import answer_prompt
    try:
        result = answer_prompt(
            args.prompt,
            rag=not args.no_rag,
            judge=not args.no_judge,
            think=args.think,
            reground=True if args.reground else None,
            top_k=args.top_k,
        )
    except OllamaError as exc:
        print(f"error: Ollama call failed: {exc}", file=sys.stderr)
        print("  Is Ollama running and the model pulled? (ollama list)", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if result.get("reasoning"):
        print("REASONING:\n" + result["reasoning"] + "\n")
    print("ANSWER:\n" + (result.get("answer") or ""))
    print()
    print(f"[rag_enabled={result['rag_enabled']} grounded={result['grounded']} "
          f"reground_applied={result['reground_applied']} latency={result['latency_ms']}ms "
          f"call_id={result.get('call_id')}]")
    verdict = result.get("verdict")
    if verdict:
        print(f"JUDGE: {verdict['rating']} ({verdict['confidence']}) — {verdict['reason']}")
        if verdict["failure_tags"]:
            print(f"  tags: {verdict['failure_tags']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
