"""Prepare v3 datasets (modern replication) into the same processed format
as ChaosNLI (see prepare.py): one JSON line per item with a human label
distribution, entropy, and entropy-quintile bucket.

Datasets:
  dices — DICES-350 (Google, CC BY 4.0): 350 conversation-safety items,
          123 raters each (104 after the README's removed_raters_350 filter,
          applied here to match published work). Label = Q_overall, the
          3-way "is the last chatbot response unsafe?" answer.
          Labels: ["no", "yes", "unsure"]  (yes = unsafe).
  par   — LeWiDi-2025 Paraphrase: 500 QQP question pairs, the same 4
          annotators each, Likert -5..+5 binned to 3 classes
          (negative -> "not_paraphrase", 0 -> "unclear", positive ->
          "paraphrase"); raw 11-bin soft label kept for AWD comparability.
          GATE-SCOPE ONLY (4 annotators cannot support the router split).

Usage: python scripts/prepare_v3.py dices|par
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

REMOVED_RATERS_350 = {
    "297514565398139", "297515609163939", "297515750682315", "297515617432733",
    "297541515566649", "297541515769980", "297515629971478", "297059995361243",
    "297541522412126", "297540556928761", "297541321453321", "297540562350921",
    "297540983991638", "297060365288109", "297514543980607", "297515729806999",
    "297541271027233", "296709611112092", "296709543131761",
}


def entropy(dist: list[float]) -> float:
    return -sum(p * math.log2(p) for p in dist if p > 0)


def quintile_buckets(rows: list[dict]) -> None:
    """Assign entropy_bucket 0-4 by within-dataset entropy quintiles."""
    ents = sorted(r["human_entropy"] for r in rows)
    n = len(ents)
    edges = [ents[int(n * q / 5)] for q in range(1, 5)]
    for r in rows:
        b = sum(r["human_entropy"] >= e for e in edges)
        r["entropy_bucket"] = b
    print(f"quintile edges: {[round(e, 3) for e in edges]}")


def write(rows: list[dict], name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    per_bucket = Counter(r["entropy_bucket"] for r in rows)
    print(f"{name}: {len(rows)} items -> {path}")
    print(f"items per bucket: {dict(sorted(per_bucket.items()))}")
    mean_ent = sum(r["human_entropy"] for r in rows) / len(rows)
    print(f"mean human entropy: {mean_ent:.3f} bits")


def prepare_dices() -> None:
    labels = ["no", "yes", "unsure"]
    per_item: dict[str, list[str]] = defaultdict(list)
    meta: dict[str, dict] = {}
    n_removed = 0
    with (RAW / "dices" / "dices350.csv").open() as f:
        for row in csv.DictReader(f):
            if row["rater_id"] in REMOVED_RATERS_350:
                n_removed += 1
                continue
            item = row["item_id"]
            per_item[item].append(row["Q_overall"].strip().lower())
            if item not in meta:
                meta[item] = {
                    "context": row["context"],
                    "response": row["response"],
                    "safety_gold": row["safety_gold"],
                    "degree_of_harm": row["degree_of_harm"],
                }
    print(f"dices: {len(per_item)} items, {n_removed} ratings removed by rater filter")

    rows = []
    for item, votes in per_item.items():
        assert all(v in labels for v in votes), f"unexpected label in {item}: {set(votes)}"
        counts = [votes.count(l) for l in labels]
        total = sum(counts)
        dist = [c / total for c in counts]
        rows.append({
            "uid": f"dices-{item}",
            "dataset": "dices350",
            "context": meta[item]["context"],
            "response": meta[item]["response"],
            "human_dist": dist,
            "human_counts": counts,
            "n_annotators": total,
            "human_entropy": entropy(dist),
            "majority_label": labels[counts.index(max(counts))],
            "safety_gold": meta[item]["safety_gold"].strip().lower(),
            "degree_of_harm": meta[item]["degree_of_harm"],
        })
    n_ann = Counter(r["n_annotators"] for r in rows)
    print(f"annotators per item: {dict(sorted(n_ann.items()))}")
    rows.sort(key=lambda r: r["uid"])
    quintile_buckets(rows)
    write(rows, "dices350")


def prepare_par() -> None:
    labels = ["not_paraphrase", "unclear", "paraphrase"]

    def bin3(rating: int) -> str:
        return labels[0] if rating < 0 else (labels[1] if rating == 0 else labels[2])

    rows = []
    for split in ["train", "dev", "test"]:
        data = json.loads((RAW / "lewidi" / f"Paraphrase_{split}.json").read_text())
        for item_id, item in data.items():
            ratings = [int(v) for v in item["annotations"].values()]
            assert len(ratings) == 4, f"{item_id}: {len(ratings)} annotations"
            votes = [bin3(r) for r in ratings]
            counts = [votes.count(l) for l in labels]
            dist = [c / 4 for c in counts]
            rows.append({  # entropy_bucket set below as a DISCRETE level, not quintile
                "uid": f"par-{item_id}",
                "dataset": "lewidi-par",
                "question1": item["text"]["Question1"],
                "question2": item["text"]["Question2"],
                "human_dist": dist,
                "human_counts": counts,
                "n_annotators": 4,
                "human_entropy": entropy(dist),
                "majority_label": labels[counts.index(max(counts))],
                "raw_ratings": ratings,
                "soft_label_11": item["soft_label"],
                "lewidi_split": split,
            })
    rows.sort(key=lambda r: r["uid"])
    # 4 votes over 3 bins admit only 4 entropy values (4-0-0: 0.0, 3-1-0:
    # 0.811, 2-2-0: 1.0, 2-1-1: 1.5 bits) — quintiles are meaningless, so
    # bucket by the discrete level itself.
    levels = sorted({round(r["human_entropy"], 6) for r in rows})
    print(f"discrete entropy levels: {levels}")
    for r in rows:
        r["entropy_bucket"] = levels.index(round(r["human_entropy"], 6))
    write(rows, "par")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("dices", "par"):
        sys.exit("usage: prepare_v3.py dices|par")
    (prepare_dices if sys.argv[1] == "dices" else prepare_par)()
