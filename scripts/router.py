"""v2: the triage router budget simulation (design doc section 6, revised).

Revised after adversarial code review + methods review (2026-08-19):
  - regret oracles added (expected-gain = the fair upper bound; realized-gain
    = clairvoyant skyline); entropy-oracle demoted to a finding, not a bound
  - k swept (1..50) so budget = total annotations, testing breadth vs depth
  - channel-disagreement signal added: JSD(sampled, verbalized)
  - split-noise floor reported; bootstrap-pool protocol as robustness
  - capture ratios replaced by AUC-based capture + budget-to-match
  - item-cluster bootstrap CIs on key strategy gaps
  - TVD as robustness metric

Protocol (default "split"): each item's 100 annotations are split 50/50 into
TRUTH (target distribution) and POOL (source of simulated annotations) —
disjoint by construction, no double use. A routed item receives k annotations
drawn without replacement from its POOL; unrouted items keep the model's
distribution (--base sampled | verb). Robustness protocol "bootstrap":
truth = all 100 counts, draws with replacement (optimistic bias, stated).

Strategies (each routes the top-B fraction by signal, random tie-breaks):
  random          uniform order
  selfcons        self-consistency entropy (CoAnnotating-style; mostly ties)
  verbalized      entropy of the verbalized distribution estimate
  chandis         JSD(sampled dist, verbalized dist) — channel disagreement
  entropy_oracle  true (TRUTH-half) entropy — best possible *entropy* signal
  egain_oracle    expected gain jsd_base - E[jsd_human_k] (MC) — fair bound
  rgain_oracle    realized gain (clairvoyant skyline)

Usage:
  python scripts/router.py --split examples --tag examples
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["entailment", "neutral", "contradiction"]

K_SWEEP = [1, 3, 5, 10, 20, 50]
K_MAIN = 10
N_REPLICATES = 20
N_MC = 200            # Monte-Carlo redraws for the expected-gain oracle
FRACS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SEED = 20260819
KEY_FRACS = [0.1, 0.2, 0.3]
N_ITEM_BOOT = 1000

STRATEGIES = ["random", "selfcons", "verbalized", "chandis",
              "entropy_oracle", "egain_oracle", "rgain_oracle"]
STRAT_LABELS = {
    "random": "Random",
    "selfcons": "Self-consistency entropy",
    "verbalized": "Verbalized entropy",
    "chandis": "Channel disagreement JSD(samp, verb)",
    "entropy_oracle": "Entropy oracle (true entropy)",
    "egain_oracle": "Expected-gain oracle (fair bound)",
    "rgain_oracle": "Realized-gain skyline",
}
PLOT_STRATS = ["random", "selfcons", "verbalized", "chandis",
               "entropy_oracle", "egain_oracle"]
COLORS = {"random": "#52514e", "selfcons": "#eb6834", "verbalized": "#2a78d6",
          "chandis": "#1baf7a", "entropy_oracle": "#4a3aa7",
          "egain_oracle": "#0b0b0b", "rgain_oracle": "#e87ba4"}
DASHES = {"entropy_oracle": (4, 2), "egain_oracle": (1, 1)}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "figure.dpi": 150,
})


# ---------- vectorized metrics on (..., 3) arrays ----------

def _kl_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    ratio = np.where(p > 0, p / np.maximum(q, 1e-300), 1.0)
    return np.sum(np.where(p > 0, p * np.log2(ratio), 0.0), axis=-1)


def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * _kl_bits(p, m) + 0.5 * _kl_bits(q, m)


def tvd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.abs(p - q), axis=-1)


def ent_bits(p: np.ndarray) -> np.ndarray:
    return -np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-300)), 0.0), axis=-1)


# ---------- data ----------

def load(split: str, tag: str) -> list[dict]:
    examples = {json.loads(l)["uid"]: json.loads(l)
                for l in (ROOT / "data" / "processed" / f"{split}.jsonl").read_text().splitlines()}
    sampled = {json.loads(l)["uid"]: json.loads(l)["model_dist"]
               for l in (ROOT / "results" / f"model_dist_{tag}.jsonl").read_text().splitlines()}
    verb = {json.loads(l)["uid"]: json.loads(l)["verb_dist"]
            for l in (ROOT / "results" / f"verbalized_{split}.jsonl").read_text().splitlines()}
    items, skipped = [], 0
    for uid, ex in examples.items():
        if uid not in sampled or uid not in verb:
            skipped += 1
            continue
        items.append({
            "uid": uid,
            "dataset": ex["dataset"],
            "counts": np.array(ex["human_counts"], dtype=np.int64),
            "sampled": np.array(sampled[uid]),
            "verb": np.array(verb[uid]),
        })
    if skipped:
        print(f"[load] WARNING: skipped {skipped} items missing from a results file")
    return items


# ---------- one replicate of the world ----------

def draw_world(items: list[dict], rng: np.random.Generator, protocol: str) -> dict:
    """Truth distributions, per-k human estimates, and per-k expected human JSD."""
    n = len(items)
    truth = np.empty((n, 3))
    pool_dist = np.empty((n, 3))
    human = {k: np.empty((n, 3)) for k in K_SWEEP}
    e_jsd_human = {k: np.empty(n) for k in K_SWEEP}
    for i, it in enumerate(items):
        if protocol == "split":
            truth_counts = rng.multivariate_hypergeometric(it["counts"], 50)
            pool = it["counts"] - truth_counts
            truth[i] = truth_counts / 50.0
            pool_dist[i] = pool / 50.0
            for k in K_SWEEP:
                human[k][i] = rng.multivariate_hypergeometric(pool, k) / k
                mc = rng.multivariate_hypergeometric(pool, k, size=N_MC) / k
                e_jsd_human[k][i] = jsd(mc, truth[i]).mean()
        else:  # bootstrap protocol: truth = all 100; draws with replacement
            p = it["counts"] / 100.0
            truth[i] = p
            pool_dist[i] = p
            for k in K_SWEEP:
                human[k][i] = rng.multinomial(k, p) / k
                mc = rng.multinomial(k, p, size=N_MC) / k
                e_jsd_human[k][i] = jsd(mc, truth[i]).mean()
    return {"truth": truth, "pool_dist": pool_dist, "human": human,
            "e_jsd_human": e_jsd_human}


# ---------- simulation ----------

def simulate(items: list[dict], base_key: str, protocol: str, metric) -> dict:
    n = len(items)
    base = np.stack([it[base_key] for it in items])
    sampled = np.stack([it["sampled"] for it in items])
    verb = np.stack([it["verb"] for it in items])
    selfcons_sig = ent_bits(sampled)
    verb_sig = ent_bits(verb)
    chandis_sig = jsd(sampled, verb)

    n_route = [int(round(f * n)) for f in FRACS]
    curves = {(s, k): np.empty((len(FRACS), N_REPLICATES)) for s in STRATEGIES for k in K_SWEEP}
    floors, gap_samples = [], {f: [] for f in KEY_FRACS}

    for rep in range(N_REPLICATES):
        rng = np.random.default_rng(SEED + rep)
        world = draw_world(items, rng, protocol)
        truth = world["truth"]
        jsd_base = metric(base, truth)
        # split-noise floor: an independent 50-annotation sample vs the truth half
        floors.append(float(metric(world["pool_dist"], truth).mean())
                      if protocol == "split" else 0.0)

        tiebreak = rng.random(n)
        for k in K_SWEEP:
            jsd_human = metric(world["human"][k], truth)
            egain_sig = jsd_base - world["e_jsd_human"][k]
            rgain_sig = jsd_base - jsd_human
            orders = {
                "random": rng.permutation(n),
                "selfcons": np.lexsort((tiebreak, -selfcons_sig)),
                "verbalized": np.lexsort((tiebreak, -verb_sig)),
                "chandis": np.lexsort((tiebreak, -chandis_sig)),
                "entropy_oracle": np.lexsort((tiebreak, -ent_bits(truth))),
                "egain_oracle": np.lexsort((tiebreak, -egain_sig)),
                "rgain_oracle": np.lexsort((tiebreak, -rgain_sig)),
            }
            total_base = jsd_base.sum()
            routed_at = {}
            for s in STRATEGIES:
                order = orders[s]
                gain = jsd_human[order] - jsd_base[order]
                cum = np.concatenate([[0.0], np.cumsum(gain)])
                for fi, m in enumerate(n_route):
                    curves[(s, k)][fi, rep] = (total_base + cum[m]) / n
                routed_at[s] = order

            # item-cluster bootstrap samples for verbalized-random gap (k main only)
            if k == K_MAIN:
                for f in KEY_FRACS:
                    m = n_route[FRACS.tolist().index(f)]
                    vals = {}
                    for s in ("random", "verbalized"):
                        picked = np.zeros(n, dtype=bool)
                        picked[routed_at[s][:m]] = True
                        vals[s] = np.where(picked, jsd_human, jsd_base)
                    gap_samples[f].append(vals["verbalized"] - vals["random"])

    # ---- assemble output ----
    out = {"base": base_key, "protocol": protocol, "n_items": n,
           "replicates": N_REPLICATES, "fracs": FRACS.tolist(),
           "k_sweep": K_SWEEP, "k_main": K_MAIN,
           "split_noise_floor": float(np.mean(floors)) if protocol == "split" else None,
           "curves": {}, "auc": {}, "capture_auc": {}, "gap_ci": {}}
    for (s, k), arr in curves.items():
        out["curves"][f"{s}@k{k}"] = {
            "mean": arr.mean(axis=1).tolist(),
            "lo": np.quantile(arr, 0.025, axis=1).tolist(),
            "hi": np.quantile(arr, 0.975, axis=1).tolist(),
        }
    # AUC over budget fraction (k main), capture normalized by expected-gain oracle
    for s in STRATEGIES:
        mean_curve = np.array(out["curves"][f"{s}@k{K_MAIN}"]["mean"])
        out["auc"][s] = float(np.trapezoid(mean_curve, FRACS))
    a_r, a_e = out["auc"]["random"], out["auc"]["egain_oracle"]
    for s in STRATEGIES:
        out["capture_auc"][s] = float((a_r - out["auc"][s]) / (a_r - a_e)) if a_r > a_e else None

    # item-cluster bootstrap CI on the verbalized-random gap
    rng = np.random.default_rng(SEED + 777)
    for f in KEY_FRACS:
        per_rep = np.stack(gap_samples[f])          # (reps, n) per-item gaps
        mean_item_gap = per_rep.mean(axis=0)        # average over replicates
        boots = np.empty(N_ITEM_BOOT)
        for b in range(N_ITEM_BOOT):
            idx = rng.integers(0, n, size=n)
            boots[b] = mean_item_gap[idx].mean()
        out["gap_ci"][str(f)] = {
            "mean": float(mean_item_gap.mean()),
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)),
        }
    return out


# ---------- reporting ----------

def budget_to_match(res: dict, f_target: float, s: str = "verbalized") -> float | None:
    """Annotation multiple random needs to match strategy s at budget f_target."""
    fi = res["fracs"].index(f_target)
    target = res["curves"][f"{s}@k{res['k_main']}"]["mean"][fi]
    rand = np.array(res["curves"][f"random@k{res['k_main']}"]["mean"])
    below = np.where(rand <= target)[0]
    if len(below) == 0:
        return None
    f_match = np.interp(target, rand[::-1], np.array(res["fracs"])[::-1])
    return float(f_match / f_target) if f_target > 0 else None


def plot_main(res: dict, tag: str) -> None:
    k = res["k_main"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.array(res["fracs"]) * 100
    for s in PLOT_STRATS:
        c = res["curves"][f"{s}@k{k}"]
        ax.plot(x, c["mean"], color=COLORS[s], lw=2, label=STRAT_LABELS[s],
                dashes=DASHES.get(s, (None, None)) if s in DASHES else "")
        ax.fill_between(x, c["lo"], c["hi"], color=COLORS[s], alpha=0.10, lw=0)
    if res["split_noise_floor"]:
        ax.axhline(res["split_noise_floor"], color=INK2, lw=1, ls=":",
                   alpha=0.7)
        ax.annotate("50v50 split-noise floor", (x[-1], res["split_noise_floor"]),
                    ha="right", va="bottom", fontsize=8, color=INK2)
    base_name = "sampled" if res["base"] == "sampled" else "verbalized"
    ax.set_xlabel(f"Items routed to humans (%), k={k} annotations each")
    ax.set_ylabel("Mean JSD to true label distribution")
    ax.set_title(f"Router value by strategy (unrouted base: {base_name} distribution)")
    ax.grid(color=INK2, alpha=0.15, lw=0.5)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"router_{res['base']}_{tag}.png")
    plt.close(fig)


def plot_k_sweep(res: dict, tag: str) -> None:
    """Breadth vs depth: verbalized strategy at each k, x = total annotations/item."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fr = np.array(res["fracs"])
    shades = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    for k, col in zip(res["k_sweep"], shades):
        c = res["curves"][f"verbalized@k{k}"]
        ax.plot(fr * k, c["mean"], color=col, lw=2, label=f"k={k}")
    c = res["curves"][f"random@k{res['k_main']}"]
    ax.plot(fr * res["k_main"], c["mean"], color="#52514e", lw=1.5, ls="--",
            label=f"random, k={res['k_main']}")
    ax.set_xlabel("Total human annotations per dataset item (budget)")
    ax.set_ylabel("Mean JSD to true label distribution")
    ax.set_title(f"Breadth vs depth (verbalized routing, base: {res['base']})")
    ax.set_xlim(0, 20)
    ax.grid(color=INK2, alpha=0.15, lw=0.5)
    ax.legend(frameon=False, fontsize=9, title="annotations per routed item")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"router_ksweep_{res['base']}_{tag}.png")
    plt.close(fig)


def report(res: dict) -> None:
    k = res["k_main"]
    print(f"== base: {res['base']}  protocol: {res['protocol']}  "
          f"(n={res['n_items']}, k_main={k}) ==")
    header = f"{'budget':>8}" + "".join(f"{STRAT_LABELS[s][:14]:>16}" for s in PLOT_STRATS)
    print(header)
    for f in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        fi = res["fracs"].index(f)
        row = "".join(f"{res['curves'][f'{s}@k{k}']['mean'][fi]:>16.4f}" for s in PLOT_STRATS)
        print(f"{f:>7.0%} {row}")
    print("\nAUC capture vs expected-gain oracle (k=10): "
          + ", ".join(f"{s}={res['capture_auc'][s]:.1%}" for s in
                      ("selfcons", "verbalized", "chandis", "entropy_oracle")
                      if res["capture_auc"][s] is not None))
    for f in KEY_FRACS:
        g = res["gap_ci"][str(f)]
        print(f"verbalized-random gap @ {f:.0%} budget: {g['mean']:+.4f} "
              f"[item-bootstrap 95% CI {g['lo']:+.4f}, {g['hi']:+.4f}]")
    for f in KEY_FRACS:
        m = budget_to_match(res, f)
        if m:
            print(f"budget-to-match @ {f:.0%}: random needs {m:.2f}x the annotations")
    if res["split_noise_floor"]:
        print(f"split-noise floor (100-count dist vs truth half): {res['split_noise_floor']:.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="examples")
    parser.add_argument("--tag", default="examples")
    parser.add_argument("--base", choices=["sampled", "verb", "both"], default="both")
    parser.add_argument("--protocol", choices=["split", "bootstrap"], default="split")
    parser.add_argument("--metric", choices=["jsd", "tvd"], default="jsd")
    args = parser.parse_args()

    metric = jsd if args.metric == "jsd" else tvd
    items = load(args.split, args.tag)
    print(f"router simulation: {len(items)} items, k sweep {K_SWEEP}, "
          f"{N_REPLICATES} replicates, {len(FRACS)} budget points, "
          f"metric={args.metric}, protocol={args.protocol}\n")

    bases = ["sampled", "verb"] if args.base == "both" else [args.base]
    for base_key in bases:
        res = simulate(items, base_key, args.protocol, metric)
        suffix = f"_{args.metric}" if args.metric != "jsd" else ""
        suffix += f"_{args.protocol}" if args.protocol != "split" else ""
        (ROOT / "results" / f"router_{base_key}_{args.split}{suffix}.json").write_text(
            json.dumps(res, indent=2))
        if args.metric == "jsd" and args.protocol == "split":
            plot_main(res, args.split)
            plot_k_sweep(res, args.split)
        report(res)


if __name__ == "__main__":
    main()
