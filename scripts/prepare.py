"""Prepare ChaosNLI for the stratified agreement study.

Reads the raw ChaosNLI jsonl files, normalizes them into one dataset with
per-example human label distributions, entropy, and entropy-quintile buckets,
and writes data/processed/examples.jsonl plus a fixed pilot subset.

ChaosNLI label_dist / label_count order is [e, n, c] for SNLI/MNLI and
[1, 2] (hypothesis choice) for abductive NLI. v0 uses SNLI + MNLI only,
so every example has the 3-way label space.
"""

import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "chaosNLI_v1.0"
OUT = ROOT / "data" / "processed"

LABELS = ["entailment", "neutral", "contradiction"]  # matches [e, n, c] order
SHORT = {"e": "entailment", "n": "neutral", "c": "contradiction"}

PILOT_SIZE = 200
SEED = 20260805


def load_split(path: Path, source_name: str) -> list[dict]:
    examples = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        dist = rec["label_dist"]
        counts = rec["label_count"]
        assert abs(sum(dist) - 1.0) < 1e-6 and sum(counts) == 100
        examples.append({
            "uid": rec["uid"],
            "dataset": source_name,
            "premise": rec["example"]["premise"],
            "hypothesis": rec["example"]["hypothesis"],
            "human_dist": dist,                      # [e, n, c]
            "human_counts": counts,
            "human_entropy": rec["entropy"],          # precomputed, base 2
            "majority_label": SHORT[rec["majority_label"]],
            "old_gold_label": SHORT[rec["old_label"]],  # original single gold label
        })
    return examples


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    examples = load_split(RAW / "chaosNLI_snli.jsonl", "snli")
    examples += load_split(RAW / "chaosNLI_mnli_m.jsonl", "mnli")

    # Entropy quintiles over the combined set
    entropies = np.array([ex["human_entropy"] for ex in examples])
    edges = np.quantile(entropies, [0.2, 0.4, 0.6, 0.8])
    for ex in examples:
        ex["entropy_bucket"] = int(np.searchsorted(edges, ex["human_entropy"]))

    with (OUT / "examples.jsonl").open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Pilot: stratified by bucket so all entropy levels are represented
    rng = random.Random(SEED)
    pilot = []
    per_bucket = PILOT_SIZE // 5
    for b in range(5):
        bucket = [ex for ex in examples if ex["entropy_bucket"] == b]
        pilot += rng.sample(bucket, per_bucket)
    with (OUT / "pilot.jsonl").open("w") as f:
        for ex in pilot:
            f.write(json.dumps(ex) + "\n")

    print(f"total examples: {len(examples)} (snli+mnli)")
    print(f"entropy quintile edges: {edges.round(3).tolist()}")
    counts = np.bincount([ex["entropy_bucket"] for ex in examples], minlength=5)
    print(f"bucket sizes: {counts.tolist()}")
    print(f"pilot size: {len(pilot)} (stratified, seed={SEED})")


if __name__ == "__main__":
    main()
