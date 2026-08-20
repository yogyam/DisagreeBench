"""Verbalized-confidence elicitation (design doc section 3; v2 routing signal).

One request per example: the model outputs a percentage distribution over the
three labels directly ("imagine 100 careful annotators").

Two modes:
  run     — synchronous with a thread pool (fine for the 200-item pilot)
  submit / status / collect — Message Batches API (50% price; use for the
            full 3,113-item set)

Outputs results/verbalized_{split}.jsonl with a normalized verb_dist per
example in [e, n, c] order.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
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


def make_client() -> anthropic.Anthropic:
    load_dotenv(ROOT / ".env", override=True)
    return anthropic.Anthropic()


def load_examples(split: str) -> list[dict]:
    path = ROOT / "data" / "processed" / f"{split}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()]


def params_for(ex: dict) -> MessageCreateParamsNonStreaming:
    return MessageCreateParamsNonStreaming(
        model=MODEL,
        max_tokens=128,
        thinking={"type": "disabled"},
        output_config={"format": OUTPUT_FORMAT},
        messages=[{"role": "user", "content": PROMPT.format(
            premise=ex["premise"], hypothesis=ex["hypothesis"])}],
    )


def parse_dist(text: str) -> list[float]:
    raw = json.loads(text)
    total = sum(max(0, raw[l]) for l in LABELS) or 1
    return [max(0, raw[l]) / total for l in LABELS]


def cmd_run(args: argparse.Namespace) -> None:
    client = make_client()
    examples = load_examples(args.split)

    def one(ex: dict) -> dict:
        resp = client.messages.create(**params_for(ex))
        text = next(b.text for b in resp.content if b.type == "text")
        return {"uid": ex["uid"], "verb_dist": parse_dist(text)}

    rows, failed = [], 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, ex): ex["uid"] for ex in examples}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                failed += 1
                print(f"  failed {futures[fut]}: {e}", flush=True)
            done = len(rows) + failed
            if done % 100 == 0:
                print(f"progress: {done}/{len(examples)}", flush=True)
    _write(rows, args.split, failed)


def cmd_submit(args: argparse.Namespace) -> None:
    client = make_client()
    examples = load_examples(args.split)
    requests, meta = [], {"model": MODEL, "split": args.split, "items": {}}
    for i, ex in enumerate(examples):
        custom_id = f"v{i:05d}"
        meta["items"][custom_id] = ex["uid"]
        requests.append(Request(custom_id=custom_id, params=params_for(ex)))
    print(f"submitting {len(requests)} verbalized requests ({args.split})...")
    batch = client.messages.batches.create(requests=requests)
    meta["batch_id"] = batch.id
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"batch_verbalized_{args.split}.json").write_text(json.dumps(meta, indent=2))
    print(f"batch id: {batch.id}  status: {batch.processing_status}")


def cmd_status(args: argparse.Namespace) -> None:
    client = make_client()
    meta = json.loads((RESULTS / f"batch_verbalized_{args.split}.json").read_text())
    batch = client.messages.batches.retrieve(meta["batch_id"])
    print(f"status: {batch.processing_status}  counts: {batch.request_counts}")


def cmd_collect(args: argparse.Namespace) -> None:
    client = make_client()
    meta = json.loads((RESULTS / f"batch_verbalized_{args.split}.json").read_text())
    batch = client.messages.batches.retrieve(meta["batch_id"])
    if batch.processing_status != "ended":
        print(f"batch not finished yet: {batch.processing_status}")
        return
    rows, failed = [], 0
    for result in client.messages.batches.results(meta["batch_id"]):
        uid = meta["items"][result.custom_id]
        if result.result.type != "succeeded":
            failed += 1
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            rows.append({"uid": uid, "verb_dist": parse_dist(text)})
        except (json.JSONDecodeError, KeyError):
            failed += 1
    _write(rows, args.split, failed)


def _write(rows: list[dict], split: str, failed: int) -> None:
    with (RESULTS / f"verbalized_{split}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"verbalized: {len(rows)} ok, {failed} failed -> results/verbalized_{split}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in [("run", cmd_run), ("submit", cmd_submit),
                     ("status", cmd_status), ("collect", cmd_collect)]:
        p = sub.add_parser(name)
        p.add_argument("--split", default="pilot")
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
