"""Robustness checks: prompt-paraphrase stability + verbalized confidence.

Compares, on the 200-example pilot:
  - sampled distributions under three prompt paraphrases
    (tags: pilot, pilot-para1, pilot-para2) — how much do entropy estimates,
    determinism, and the gate move with wording? (design doc section 9)
  - verbalized percentage distributions (results/verbalized_pilot.jsonl) vs
    sampled distributions and vs human distributions — including verbalized
    entropy as an alternative routing signal (design doc section 3)

Usage: python scripts/robustness.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as shannon_entropy
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["entailment", "neutral", "contradiction"]
PARAPHRASE_TAGS = ["pilot", "pilot-para1", "pilot-para2"]


def load_dists(path: Path, key: str) -> dict[str, np.ndarray]:
    return {r["uid"]: np.array(r[key])
            for r in map(json.loads, path.read_text().splitlines())}


def ent(d: np.ndarray) -> float:
    return float(shannon_entropy(d, base=2))


def main() -> None:
    pilot = {json.loads(l)["uid"]: json.loads(l)
             for l in (ROOT / "data" / "processed" / "pilot.jsonl").read_text().splitlines()}
    uids = sorted(pilot)
    human = {u: np.array(pilot[u]["human_dist"]) for u in uids}
    h_ent = np.array([pilot[u]["human_entropy"] for u in uids])

    # ---- Paraphrase stability ----
    dists = {}
    for tag in PARAPHRASE_TAGS:
        p = ROOT / "results" / f"model_dist_{tag}.jsonl"
        if p.exists():
            dists[tag] = load_dists(p, "model_dist")
        else:
            print(f"[skip] {p.name} not found yet")
    if len(dists) > 1:
        print("== Prompt-paraphrase stability (200 pilot examples) ==")
        print(f"{'prompt':<14}{'determinism':>12}{'mean JSD vs human':>19}{'gate rho':>10}")
        for tag, d in dists.items():
            det = np.mean([max(d[u]) == 1.0 for u in uids])
            jsd = np.mean([jensenshannon(human[u], d[u], base=2) ** 2 for u in uids])
            rho = spearmanr(h_ent, [ent(d[u]) for u in uids]).statistic
            print(f"{tag:<14}{det:>11.0%}{jsd:>19.3f}{rho:>10.3f}")
        print("\npairwise agreement between prompts:")
        for a, b in combinations(dists, 2):
            da, db = dists[a], dists[b]
            modal_agree = np.mean([np.argmax(da[u]) == np.argmax(db[u]) for u in uids])
            ent_rho = spearmanr([ent(da[u]) for u in uids],
                                [ent(db[u]) for u in uids]).statistic
            jsd_ab = np.mean([jensenshannon(da[u], db[u], base=2) ** 2 for u in uids])
            print(f"  {a} vs {b}: modal-label agreement {modal_agree:.1%}, "
                  f"entropy Spearman {ent_rho:.3f}, mean JSD {jsd_ab:.3f}")

    # ---- Verbalized confidence ----
    vp = ROOT / "results" / "verbalized_pilot.jsonl"
    if vp.exists() and "pilot" in dists:
        verb = load_dists(vp, "verb_dist")
        samp = dists["pilot"]
        v_ent = np.array([ent(verb[u]) for u in uids])
        s_ent = np.array([ent(samp[u]) for u in uids])
        print("\n== Verbalized confidence (single request/example) ==")
        print(f"mean JSD vs human:   verbalized {np.mean([jensenshannon(human[u], verb[u], base=2)**2 for u in uids]):.3f}"
              f"  |  sampled {np.mean([jensenshannon(human[u], samp[u], base=2)**2 for u in uids]):.3f}")
        print(f"mean entropy:        verbalized {v_ent.mean():.3f}  |  sampled {s_ent.mean():.3f}"
              f"  |  human {h_ent.mean():.3f}")
        rho_v = spearmanr(h_ent, v_ent)
        rho_s = spearmanr(h_ent, s_ent)
        print(f"GATE (vs human entropy):  verbalized rho = {rho_v.statistic:.3f} (p={rho_v.pvalue:.1e})"
              f"  |  sampled rho = {rho_s.statistic:.3f} (p={rho_s.pvalue:.1e})")
        print(f"verbalized-vs-sampled entropy Spearman: "
              f"{spearmanr(v_ent, s_ent).statistic:.3f}")


if __name__ == "__main__":
    main()
