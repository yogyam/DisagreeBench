"""Verbalized-confidence elicitation (design doc section 3, comparison arm).

One request per example: the model outputs a percentage distribution over the
three labels directly. Runs synchronously with a small thread pool (200
requests, ~2 min). Output: results/verbalized_pilot.jsonl with a normalized
verb_dist per example in [e, n, c] order.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["entailment", "neutral", "contradiction"]
MODEL = "claude-sonnet-5"

PROMPT = """You are annotating a Natural Language Inference (NLI) example.

Premise: {premise}

Hypothesis: {hypothesis}

Imagine 100 careful human annotators labeling this example as entailment,
neutral, or contradiction. Estimate how many would choose each label.
The three numbers must sum to 100."""

OUTPUT_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {l: {"type": "integer"} for l in LABELS},
        "required": LABELS,
        "additionalProperties": False,
    },
}


def one(client: anthropic.Anthropic, ex: dict) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=128,
        thinking={"type": "disabled"},
        output_config={"format": OUTPUT_FORMAT},
        messages=[{"role": "user", "content": PROMPT.format(
            premise=ex["premise"], hypothesis=ex["hypothesis"])}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    raw = json.loads(text)
    total = sum(max(0, raw[l]) for l in LABELS) or 1
    return {"uid": ex["uid"], "verb_dist": [max(0, raw[l]) / total for l in LABELS]}


def main() -> None:
    load_dotenv(ROOT / ".env", override=True)
    client = anthropic.Anthropic()
    examples = [json.loads(l)
                for l in (ROOT / "data" / "processed" / "pilot.jsonl").read_text().splitlines()]
    rows, failed = [], 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, client, ex): ex["uid"] for ex in examples}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                failed += 1
                print(f"  failed {futures[fut]}: {e}")
    with (ROOT / "results" / "verbalized_pilot.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"verbalized: {len(rows)} ok, {failed} failed -> results/verbalized_pilot.jsonl")


if __name__ == "__main__":
    main()
