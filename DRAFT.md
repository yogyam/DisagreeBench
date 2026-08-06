# Where LLM Annotation Substitutes for Human Labelers — and Where It Doesn't

**Draft v0.1 — 2026-08-05.** Yogya Mehrotra.
Code, data pipeline, and figures: this repository.

## Abstract

The case for replacing human annotators with LLMs typically rests on a single
aggregate agreement number. We test the hypothesis that this number is
structurally misleading: that LLM–human agreement is high exactly where humans
already agree with each other, and collapses where they don't. Using ChaosNLI
(3,113 SNLI/MNLI items with 100 human annotations each), we elicit a label
*distribution* from claude-sonnet-5 via self-consistency sampling (10 samples
per item) and compare it to the human distribution with Jensen–Shannon
divergence (JSD), stratified by human label entropy. The hypothesis holds:
mean JSD rises monotonically from 0.07 [95% CI 0.06, 0.09] in the
lowest-entropy quintile to 0.31 [0.30, 0.32] in the highest, while the
conventional metric — accuracy against the majority label — declines far more
gently (0.94 → 0.56) and therefore flatters the model precisely on the
contested items. We then test whether the model's own uncertainty could route
hard items to humans, and find it cannot: the model is fully deterministic
across its 10 samples on 85% of items — including most items where humans
split near 50/50 — and the rank correlation between model self-consistency
entropy and human entropy is ρ = 0.164 [0.131, 0.196], decisively below any
usefully routable signal; enabling reasoning does not change this. The
contrast is stark: the human pool's inter-annotator agreement is
Krippendorff's α = 0.37; the model's agreement with itself is α = 0.90. Yet
when asked to *estimate* the annotator distribution directly, the same model
produces distributions 3× closer to the human ones (JSD 0.08) whose entropy
does clear the routing threshold (ρ = 0.354). The model's behavioral
uncertainty carries no information about human disagreement — but its
declarative estimate of human disagreement does.

## 1. Motivation

Data-labeling providers are being asked to justify themselves against claims
of the form "the model agrees with human labels 90% of the time at 1% of the
cost." That figure is an average over a distribution of examples that is
sharply bimodal in difficulty: most items in a real annotation stream are
unambiguous, while a minority are genuinely contested — and the contested
minority consumes most of a real annotation budget and drives most
disagreement in downstream training. If LLM agreement concentrates on the easy
majority, a headline agreement figure can be technically true and
operationally misleading, and the correct architecture is neither "replace
humans" nor "keep humans" but *triage*. This study measures the premise of
that argument directly, and then tests whether the triage router is buildable
from the model's own uncertainty.

## 2. Data

**ChaosNLI** (Nie et al., 2020) provides ~100 annotations per item for 4,645
development-set items from SNLI, MNLI, and Abductive NLI. We use the SNLI
(1,514) and MNLI (1,599) portions — 3,113 items sharing the 3-way
entailment/neutral/contradiction label space — and exclude Abductive NLI (a
2-way task). The 100-annotator sample gives a genuine label distribution per
item rather than a noisy 3–5-rater estimate, which is what makes entropy
stratification statistically meaningful. Items are bucketed into quintiles of
human label entropy computed from the full distribution (quintile edges: 0.66,
0.91, 1.05, 1.21 bits).

The pool is genuinely contested: nominal Krippendorff's α is 0.447 (SNLI),
0.283 (MNLI), 0.367 combined. This is by construction — ChaosNLI drew items
whose original 5-rater annotations disagreed — and it is exactly the regime
the substitution argument is quietly averaged over.

## 3. Method

**Elicitation.** claude-sonnet-5 (Anthropic), accessed via the Message Batches
API. Each item is sampled 10 times as an independent request. The label
options are presented in randomized order per sample (mitigating
option-ordering bias); output is constrained to a JSON schema over the three
labels, so every sample parses; extended thinking is disabled. The empirical
distribution over the 10 sampled labels is the model's label distribution.
Of 31,130 requests, 31,127 returned usable labels (3 errors).

**Verbalized arm.** As a comparison elicitation (pilot subset, n = 200, one
request per item), the model is asked to estimate how 100 careful annotators
would distribute over the three labels, output as integer percentages under
the same schema constraint.

**Metrics.** Primary: per-item Jensen–Shannon divergence (base 2) between the
model and human label distributions, averaged per entropy quintile. Reported
alongside, explicitly as the metric under critique: accuracy of the model's
modal label against the human majority label. The routing gate: Spearman rank
correlation between model self-consistency entropy and human entropy. All
means and the correlation carry 95% percentile bootstrap CIs (10,000
resamples). Contamination diagnostic: agreement of the modal label with the
*original* single gold label vs. with the ChaosNLI majority, per quintile.

## 4. Results

### 4.1 Agreement degrades with human disagreement — and accuracy hides it

| Human-entropy quintile | n | Mean JSD [95% CI] | Acc. vs. majority [95% CI] |
|---|---|---|---|
| Q1 (lowest) | 631 | 0.074 [0.064, 0.085] | 0.945 [0.926, 0.962] |
| Q2 | 615 | 0.178 [0.165, 0.192] | 0.805 [0.772, 0.836] |
| Q3 | 623 | 0.244 [0.232, 0.255] | 0.674 [0.637, 0.711] |
| Q4 | 622 | 0.263 [0.254, 0.273] | 0.614 [0.576, 0.653] |
| Q5 (highest) | 622 | 0.311 [0.302, 0.321] | 0.558 [0.519, 0.596] |

JSD rises monotonically by a factor of ~4.2 from Q1 to Q5
(figure: `figures/jsd_vs_entropy_examples.png`). Accuracy against the majority
declines much more slowly, and its floor is deceptive: on Q5 items the human
majority is itself close to a coin flip, so "56% accuracy" largely reflects
picking the more common side of a split — not agreement with what humans
collectively believe. The gap between the two curves is the headline finding:
**the standard metric is most flattering exactly where the model is furthest
from the human distribution.**

### 4.2 The gate fails: model uncertainty is not human uncertainty

If a model knew which items humans find hard, its uncertainty could route
those items to people. It does not. Across 10 samples the model returns the
identical label on 2,653 / 3,113 items (85%) — including 84% of Q5, where
human annotators split at more than 1.2 bits of entropy. Where the human pool
agrees at α = 0.367, the model agrees with itself at α = 0.904.

The rank correlation between model self-consistency entropy and human entropy
is ρ = 0.164 [95% CI 0.131, 0.196] (p ≈ 4e-20). The correlation is reliably
nonzero — the model is not blind to difficulty — but the CI excludes by a wide
margin any threshold (≥ ~0.3) at which uncertainty-based routing would
meaningfully beat random allocation. An ablation with adaptive thinking
enabled (§5) leaves the picture unchanged. Following the pre-registered logic
of the study design, we therefore do not proceed to the router simulation:
**the router's input signal does not exist.**

### 4.3 Verbalized confidence: the model can report disagreement it does not enact

Prompted instead to *estimate* the human distribution directly — "imagine 100
careful annotators; how many choose each label?" (one request per item) — the
model behaves entirely differently:

| Signal | Mean JSD vs. human | Mean entropy (bits) | Gate ρ vs. human entropy |
|---|---|---|---|
| Sampled (10× self-consistency) | 0.224 | 0.089 | 0.124 (n.s.) |
| Verbalized (est. of 100 annotators) | **0.081** | **0.853** | **0.354** (p ≈ 3e-7) |
| Human pool | — | 0.943 | — |

(Pilot subset, n = 200.) Verbalized distributions are ~3× closer to the human
distribution than sampled ones, their average entropy nearly matches the human
pool's, and verbalized entropy clears the pre-registered routing threshold
(ρ = 0.354 ≥ 0.3). Strikingly, the two uncertainty signals are uncorrelated
with each other (ρ = 0.028): they measure different quantities. Together with
§4.2, this yields the paper's sharpest formulation: **the model's *behavioral*
uncertainty carries no information about human disagreement, but its
*declarative* estimate of human disagreement does.** This inverts a common
assumption (including in our own study design) that verbalized probabilities
are poorly calibrated and self-consistency is the trustworthy signal. It also
conditionally revives the triage router: the routing signal exists — it just
has to be asked for, not observed.

One caveat attaches specifically to this result: ChaosNLI's label
*distributions* have been public since 2020, so the model may have absorbed
some of them in pretraining. A memorized distribution would inflate verbalized
performance in a way the gold-label diagnostic in §4.4 does not cover.
Replication on held-out or post-cutoff annotation data is required before the
verbalized result is load-bearing.

### 4.4 The result is not memorization

ChaosNLI is built on SNLI/MNLI, which are almost certainly in the model's
pretraining data; a model could look good by reciting original gold labels. The
diagnostic says otherwise: the modal label agrees with the ChaosNLI majority
*more* than with the original gold label in Q1–Q3 (e.g. 0.945 vs. 0.857 in
Q1), and the two converge in Q4–Q5 (0.614 vs. 0.637; 0.558 vs. 0.539). If the
model were reproducing memorized gold labels, the pattern would invert on
high-entropy items, where the original gold is essentially arbitrary. It does
not (figure: `figures/contamination_examples.png`).

## 5. Limitations

- **Elicitation is not the explanation (ablated).** Because thinking was
  disabled and output schema-constrained in the main run, we replicated the
  200-item pilot with adaptive thinking enabled and a 2,048-token budget.
  Determinism was unchanged (84% vs. 87%), model self-agreement was unchanged
  (α = 0.896 vs. 0.904), and the gate correlation did not improve
  (ρ = 0.066 [−0.06, 0.19], n.s. at n = 200). Notably, the model — free to
  reason at its own discretion — mostly declined to (mean 21 output
  tokens/sample vs. 11 without thinking): it does not even *perceive* these
  items as hard. The determinism result therefore survives its most obvious
  confound, though we cannot rule out that stronger forced-reasoning prompts
  would behave differently.
- **One model, one task type.** Results are scoped to claude-sonnet-5 on NLI
  ambiguity; subjectivity-driven disagreement (hate speech, irony — LeWiDi)
  may behave differently, and is planned as v3.
- **Sampling temperature is not tunable** on this model (the API rejects
  non-default sampling parameters), so the temperature-sensitivity analysis in
  the original design is moot; sampling ran at the provider default.
- **10 samples bound measurable entropy.** With 10 draws the minimum nonzero
  entropy is ~0.47 bits, coarsening the model-entropy scale. This attenuates
  the gate correlation somewhat; it cannot explain 85% exact determinism.
- **Prompt sensitivity: conclusions stable, but the sampled signal itself is
  noisy.** Re-running the pilot under two prompt paraphrases leaves every
  headline quantity essentially unchanged (determinism 80–87%; mean JSD
  0.21–0.23; gate ρ 0.09–0.12, never approaching 0.3; modal labels agree
  across prompts on 94% of items). However, *per-item* self-consistency
  entropy correlates only moderately across paraphrases (Spearman 0.38–0.46) —
  the little sampling variance that exists is substantially prompt-specific
  noise rather than a stable property of the item, a further reason it cannot
  serve as a routing signal.

## 6. Implications

For the substitution debate, the two results cut in opposite directions, and
both matter. The stratified curve says the cheap examples are the ones the
model already labels well — consistent with aggressive automation of the easy
majority. The failed gate says the obvious triage architecture — "let the
model flag what it's unsure about" — does not work off the shelf: the model's
confidence is nearly uniform and nearly total, so it cannot tell you which
items needed a human. The verbalized result (§4.3) points at the repair:
a routing signal exists, but it is a *declarative* capability that must be
elicited explicitly, not read off the model's sampling behavior — pending a
memorization-controlled replication, the budget-simulation study (router vs.
random vs. oracle allocation) is the natural next step, run on verbalized
entropy. Regardless, headline agreement numbers should be reported stratified
by inter-annotator agreement as a matter of course.

## Reproducibility

`scripts/prepare.py` → `scripts/elicit.py submit|status|collect` →
`scripts/analyze.py`. Elicitation: claude-sonnet-5, Message Batches API,
10 samples/item, randomized option order, JSON-schema output, thinking
disabled, seed 20260805 for all randomization and bootstraps. Total API cost:
≈ $14–21 (31,130 batched requests; 11.25M input / 0.35M output tokens).
