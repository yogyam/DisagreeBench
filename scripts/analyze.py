"""Part 1 analysis + the Part 2 gate, with bootstrap CIs and Krippendorff's alpha.

Joins the model label distributions (results/model_dist_{tag}.jsonl) with the
human distributions (data/processed/{split}.jsonl) and produces:

  1. figures/jsd_vs_entropy_{tag}.png   — mean JSD by human-entropy bucket with
     95% bootstrap CIs, naive accuracy-vs-majority on the same axis (section 5)
  2. figures/gate_scatter_{tag}.png     — model entropy vs human entropy scatter,
     Spearman rho with bootstrap CI (section 6 gate)
  3. figures/contamination_{tag}.png    — agreement with original gold vs with
     ChaosNLI majority, by bucket (section 9 diagnostic)
  4. a text report: per-bucket table with CIs, the gate, Krippendorff's alpha
     for the human pool (per source + combined) and for the model's 10 samples
     treated as 10 annotators

Usage: python scripts/analyze.py --tag examples --split examples
"""

import argparse
import json
from pathlib import Path

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as shannon_entropy
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["entailment", "neutral", "contradiction"]

N_BOOT = 10_000
BOOT_SEED = 20260805

# dataviz reference palette (validated): series-1 blue, series-2 orange
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "figure.dpi": 150,
})

BUCKET_NAMES = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]


def load(tag: str, split: str) -> tuple[list[dict], dict]:
    examples = {json.loads(l)["uid"]: json.loads(l)
                for l in (ROOT / "data" / "processed" / f"{split}.jsonl").read_text().splitlines()}
    rows = []
    for line in (ROOT / "results" / f"model_dist_{tag}.jsonl").read_text().splitlines():
        m = json.loads(line)
        ex = examples[m["uid"]]
        p, q = np.array(ex["human_dist"]), np.array(m["model_dist"])
        modal = LABELS[int(np.argmax(q))]
        rows.append({
            "uid": m["uid"],
            "dataset": ex["dataset"],
            "bucket": ex["entropy_bucket"],
            "human_counts": ex["human_counts"],
            "human_entropy": ex["human_entropy"],
            "model_dist": m["model_dist"],
            "n_samples": m["n_samples"],
            "model_entropy": float(shannon_entropy(q, base=2)),
            "jsd": float(jensenshannon(p, q, base=2) ** 2),  # JS divergence, not distance
            "acc_majority": modal == ex["majority_label"],
            "acc_old_gold": modal == ex["old_gold_label"],
        })
    return rows, examples


def boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean with 95% percentile bootstrap CI."""
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def boot_spearman(h: np.ndarray, m: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    rho, p = spearmanr(h, m)
    rhos = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(h), size=len(h))
        rhos[i] = spearmanr(h[idx], m[idx]).statistic
    return float(rho), float(p), float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975))


def krippendorff_alpha_from_counts(counts: np.ndarray) -> float:
    """Nominal alpha from a units x categories value-count matrix."""
    return float(krippendorff.alpha(value_counts=counts, level_of_measurement="nominal"))


def bucket_stats(rows: list[dict], key: str, rng: np.random.Generator) -> list[tuple[float, float, float]]:
    return [boot_ci(np.array([float(r[key]) for r in rows if r["bucket"] == b]), rng)
            for b in range(5)]


def fig_jsd_vs_entropy(jsd_s, acc_s, tag: str) -> None:
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for stats, color, label, dy in [(jsd_s, BLUE, "Model–human JSD", 9),
                                    (acc_s, ORANGE, "Accuracy vs. majority", -15)]:
        mean = np.array([s[0] for s in stats])
        lo = mean - np.array([s[1] for s in stats])
        hi = np.array([s[2] for s in stats]) - mean
        ax.errorbar(x, mean, yerr=[lo, hi], color=color, lw=2, marker="o", ms=7,
                    capsize=3, label=f"{label} (95% CI)")
        for xi, yi in zip(x, mean):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=9, color=color)
    ax.set_xticks(x, BUCKET_NAMES)
    ax.set_xlabel("Human label entropy (quintile)")
    ax.set_ylabel("Mean value")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=INK2, alpha=0.15, lw=0.5)
    ax.legend(frameon=False)
    ax.set_title("Agreement degrades where humans disagree — and accuracy hides it")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"jsd_vs_entropy_{tag}.png")
    plt.close(fig)


def fig_gate_scatter(rows, rho, p, lo, hi, tag: str) -> None:
    h = np.array([r["human_entropy"] for r in rows])
    m = np.array([r["model_entropy"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(h, m, s=14, color=BLUE, alpha=0.3, edgecolors="none")
    ax.set_xlabel("Human label entropy (bits)")
    ax.set_ylabel("Model self-consistency entropy (bits)")
    ax.set_title("The gate: does model uncertainty track human uncertainty?")
    ax.text(0.03, 0.95, f"Spearman ρ = {rho:.3f}  [95% CI {lo:.3f}, {hi:.3f}]",
            transform=ax.transAxes, fontsize=11, va="top")
    ax.grid(color=INK2, alpha=0.15, lw=0.5)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"gate_scatter_{tag}.png")
    plt.close(fig)


def fig_contamination(maj_s, old_s, tag: str) -> None:
    x, w = np.arange(5), 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for stats, off, color, label in [(maj_s, -w / 2, BLUE, "Agrees with ChaosNLI majority"),
                                     (old_s, w / 2, ORANGE, "Agrees with original gold label")]:
        mean = np.array([s[0] for s in stats])
        lo = mean - np.array([s[1] for s in stats])
        hi = np.array([s[2] for s in stats]) - mean
        ax.bar(x + off, mean, w - 0.02, color=color, label=label)
        ax.errorbar(x + off, mean, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    ax.set_xticks(x, BUCKET_NAMES)
    ax.set_xlabel("Human label entropy (quintile)")
    ax.set_ylabel("Agreement rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=INK2, alpha=0.15, lw=0.5)
    ax.legend(frameon=False)
    ax.set_title("Contamination check: tracking gold-from-memory vs. the crowd")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"contamination_{tag}.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="examples")
    parser.add_argument("--split", default="examples")
    args = parser.parse_args()

    (ROOT / "figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    rows, _ = load(args.tag, args.split)
    print(f"analyzed {len(rows)} examples ({args.tag}), {N_BOOT} bootstrap resamples\n")

    jsd_s = bucket_stats(rows, "jsd", rng)
    acc_s = bucket_stats(rows, "acc_majority", rng)
    old_s = bucket_stats(rows, "acc_old_gold", rng)

    print(f"{'bucket':<8}{'n':>5}{'mean JSD [95% CI]':>26}{'acc(maj) [95% CI]':>26}{'acc(old)':>10}")
    for b in range(5):
        n = sum(1 for r in rows if r["bucket"] == b)
        j, a, o = jsd_s[b], acc_s[b], old_s[b]
        print(f"{'Q' + str(b + 1):<8}{n:>5}"
              f"{f'{j[0]:.3f} [{j[1]:.3f}, {j[2]:.3f}]':>26}"
              f"{f'{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]':>26}"
              f"{o[0]:>10.3f}")

    # ---- The gate ----
    h = np.array([r["human_entropy"] for r in rows])
    m = np.array([r["model_entropy"] for r in rows])
    rho, p, lo, hi = boot_spearman(h, m, rng)
    det = sum(1 for r in rows if max(r["model_dist"]) == 1.0)
    print(f"\nGATE — Spearman(model entropy, human entropy) = {rho:.3f} "
          f"[95% CI {lo:.3f}, {hi:.3f}]  (p = {p:.2e})")
    print(f"model fully deterministic on {det}/{len(rows)} examples ({100 * det / len(rows):.0f}%)")

    # ---- Krippendorff's alpha ----
    print("\nKrippendorff's alpha (nominal):")
    for name in ["snli", "mnli"]:
        sub = np.array([r["human_counts"] for r in rows if r["dataset"] == name])
        print(f"  human pool, {name:<9}: {krippendorff_alpha_from_counts(sub):.3f}")
    all_h = np.array([r["human_counts"] for r in rows])
    print(f"  human pool, combined : {krippendorff_alpha_from_counts(all_h):.3f}")
    model_counts = np.array([(np.array(r["model_dist"]) * r["n_samples"]).round()
                             for r in rows])
    print(f"  model 10 samples as 10 annotators: {krippendorff_alpha_from_counts(model_counts):.3f}")

    fig_jsd_vs_entropy(jsd_s, acc_s, args.tag)
    fig_gate_scatter(rows, rho, p, lo, hi, args.tag)
    fig_contamination(acc_s, old_s, args.tag)
    print("\nfigures written to figures/")


if __name__ == "__main__":
    main()
