"""Elicit an LLM label distribution per example via self-consistency sampling.

Uses the Message Batches API (50% price) with claude-sonnet-5. Each example is
sampled N times; each sample is an independent batch request with the label
options presented in randomized order (option-ordering mitigation, design doc
section 9). Thinking is disabled and output is constrained to a JSON schema so
every sample is a clean label.

Usage:
  python scripts/elicit.py submit  --split pilot          # build + submit batch
  python scripts/elicit.py status  --tag pilot            # poll processing status
  python scripts/elicit.py collect --tag pilot            # fetch results, aggregate

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def make_client() -> anthropic.Anthropic:
    # override=True so the project .env beats any stale key from the shell profile
    load_dotenv(ROOT / ".env", override=True)
    return anthropic.Anthropic()

MODEL = "claude-sonnet-5"
N_SAMPLES = 10
SEED = 20260805
LABELS = ["entailment", "neutral", "contradiction"]

PROMPT_TEMPLATES = {
    "default": """You are annotating a Natural Language Inference (NLI) example.

Premise: {premise}

Hypothesis: {hypothesis}

Does the premise entail the hypothesis? Answer with exactly one label:
{options}

Give the label a careful human annotator would choose.""",
    "para1": """Consider the following pair of sentences.

Premise: {premise}

Hypothesis: {hypothesis}

Assuming the premise is true, is the hypothesis definitely true, possibly true, \
or definitely false? Pick exactly one of these labels:
{options}""",
    "para2": """Read the premise, then judge the logical relationship of the \
hypothesis to it.

Premise: {premise}

Hypothesis: {hypothesis}

Select the single best label from:
{options}

Answer as an experienced annotation worker would.""",
}

OUTPUT_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": LABELS}},
        "required": ["label"],
        "additionalProperties": False,
    },
}


def build_requests(split: str, thinking: bool, prompt_name: str) -> tuple[list[Request], dict]:
    rng = random.Random(SEED)
    examples = [json.loads(l) for l in (PROCESSED / f"{split}.jsonl").read_text().splitlines()]
    meta = {"model": MODEL, "n_samples": N_SAMPLES, "split": split,
            "thinking": thinking, "prompt": prompt_name, "items": {}}
    requests = []
    for i, ex in enumerate(examples):
        for s in range(N_SAMPLES):
            options = LABELS.copy()
            rng.shuffle(options)
            custom_id = f"ex{i:05d}-s{s:02d}"
            meta["items"][custom_id] = {"uid": ex["uid"], "option_order": options}
            prompt = PROMPT_TEMPLATES[prompt_name].format(
                premise=ex["premise"],
                hypothesis=ex["hypothesis"],
                options="\n".join(f"- {o}" for o in options),
            )
            params = dict(
                model=MODEL,
                # thinking variant: omit `thinking` (adaptive is the model default)
                # and leave room for reasoning tokens before the JSON answer
                max_tokens=2048 if thinking else 64,
                output_config={"format": OUTPUT_FORMAT},
                messages=[{"role": "user", "content": prompt}],
            )
            if not thinking:
                params["thinking"] = {"type": "disabled"}
            requests.append(Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(**params),
            ))
    return requests, meta


def cmd_submit(args: argparse.Namespace) -> None:
    client = make_client()
    tag = args.tag or (args.split
                       + ("-thinking" if args.thinking else "")
                       + (f"-{args.prompt}" if args.prompt != "default" else ""))
    requests, meta = build_requests(args.split, args.thinking, args.prompt)
    variant = "thinking" if args.thinking else "no-thinking"
    print(f"submitting {len(requests)} requests ({args.split}, {N_SAMPLES} samples/example, "
          f"{variant}, prompt={args.prompt})...")
    batch = client.messages.batches.create(requests=requests)
    meta["batch_id"] = batch.id
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"batch_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"batch id: {batch.id}  status: {batch.processing_status}")
    print(f"metadata saved to results/batch_{tag}.json")
    print(f"poll with: python scripts/elicit.py status --tag {tag}")


def cmd_status(args: argparse.Namespace) -> None:
    client = make_client()
    meta = json.loads((RESULTS / f"batch_{args.tag}.json").read_text())
    batch = client.messages.batches.retrieve(meta["batch_id"])
    print(f"status: {batch.processing_status}  counts: {batch.request_counts}")


def cmd_collect(args: argparse.Namespace) -> None:
    client = make_client()
    meta = json.loads((RESULTS / f"batch_{args.tag}.json").read_text())
    batch = client.messages.batches.retrieve(meta["batch_id"])
    if batch.processing_status != "ended":
        print(f"batch not finished yet: {batch.processing_status}")
        return

    samples, errors = [], 0
    for result in client.messages.batches.results(meta["batch_id"]):
        item = meta["items"][result.custom_id]
        if result.result.type != "succeeded":
            errors += 1
            continue
        msg = result.result.message
        if msg.stop_reason == "refusal":
            errors += 1
            continue
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            label = json.loads(text)["label"]
        except (json.JSONDecodeError, KeyError):
            errors += 1
            continue
        samples.append({
            "custom_id": result.custom_id,
            "uid": item["uid"],
            "label": label,
            "option_order": item["option_order"],
        })

    with (RESULTS / f"samples_{args.tag}.jsonl").open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Aggregate into per-example model distributions, [e, n, c] order
    by_uid: dict[str, Counter] = {}
    for s in samples:
        by_uid.setdefault(s["uid"], Counter())[s["label"]] += 1
    with (RESULTS / f"model_dist_{args.tag}.jsonl").open("w") as f:
        for uid, counts in by_uid.items():
            total = sum(counts.values())
            f.write(json.dumps({
                "uid": uid,
                "n_samples": total,
                "model_dist": [counts.get(l, 0) / total for l in LABELS],
            }) + "\n")

    print(f"collected {len(samples)} samples across {len(by_uid)} examples ({errors} errors/unparseable)")
    print(f"wrote results/samples_{args.tag}.jsonl and results/model_dist_{args.tag}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit")
    p.add_argument("--split", default="pilot")
    p.add_argument("--thinking", action="store_true", help="adaptive thinking variant")
    p.add_argument("--prompt", default="default", choices=list(PROMPT_TEMPLATES))
    p.add_argument("--tag", default=None, help="override results tag (default: split[-thinking])")
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("status"); p.add_argument("--tag", default="pilot"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("collect"); p.add_argument("--tag", default="pilot"); p.set_defaults(fn=cmd_collect)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
