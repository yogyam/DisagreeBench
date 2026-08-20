# Where LLM Annotation Substitutes for Human Labelers — and Where It Doesn't

**Draft v0.2 — 2026-08-19.** Yogya Mehrotra.
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
produces distributions 3× closer to the human ones (JSD 0.07, full set) whose
entropy does clear the routing threshold (ρ = 0.393). The model's behavioral
uncertainty carries no information about human disagreement — but its
declarative estimate of human disagreement does. Finally, we run the
budget-allocation experiment this signal enables: a simulated router that
sends the top-scoring items to k human annotators and keeps the model's
distribution for the rest. Against the realistic baseline (model
self-consistency distributions), routing on verbalized entropy captures 37% of
an expected-gain oracle's achievable improvement, and random allocation needs
~1.5× the annotation budget to match it. Against the strongest baseline (the
model's own verbalized distributions), the same signal is *worse than random*:
the model's residual errors hide in items it is confidently wrong about.
Model-reported uncertainty tells you where the task is hard — not where the
model is wrong.

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

**Verbalized arm.** As a comparison elicitation (one request per item, full
3,113-item set), the model is asked to estimate how 100 careful annotators
would distribute over the three labels, output as integer percentages under
the same schema constraint.

**Router simulation.** Each item's 100 annotations are split 50/50: one half
defines the ground-truth distribution, the other is the finite pool routed
items draw from (so the router can never "buy" the evaluation target, and
even perfect routing pays annotator-sampling noise). A router scores all
items, sends the top fraction f to humans — receiving k annotations drawn
without replacement from the pool — and keeps the model's distribution for
the rest. We sweep f from 0 to 1 and k ∈ {1, 3, 5, 10, 20, 50}, with 20
replicate splits. Strategies: random; self-consistency entropy; verbalized
entropy; channel disagreement (JSD between the sampled and verbalized
distributions); an *entropy oracle* scoring by true human entropy; and two
regret oracles scoring by each item's actual improvement from routing —
*expected gain* (Monte-Carlo mean over 200 redraws; the fair upper bound,
since it knows which items benefit but not the luck of the draw) and
*realized gain* (a skyline that peeks at the draw). Headline numbers: AUC of
the budget–JSD curve, normalized as the share of the expected-gain oracle's
improvement over random ("capture"); and budget-to-match, the multiple of
annotations random allocation needs to reach a strategy's JSD at a given
budget. Item-level bootstrap (1,000 resamples) gives CIs on strategy–random
gaps. Everything is repeated with total variation distance in place of JSD
and with a bootstrap-resampling split protocol in place of the 50/50 split.

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
enabled (§5) leaves the picture unchanged. By the pre-registered logic of the
study design, self-consistency entropy is disqualified as a router input:
**this routing signal does not exist** (§4.5 confirms end-to-end that routing
on it is no better than random). The signal that does exist comes from a
different channel entirely (§4.3).

### 4.3 Verbalized confidence: the model can report disagreement it does not enact

Prompted instead to *estimate* the human distribution directly — "imagine 100
careful annotators; how many choose each label?" (one request per item) — the
model behaves entirely differently:

| Signal | Mean JSD vs. human | Mean entropy (bits) | Gate ρ vs. human entropy |
|---|---|---|---|
| Sampled (10× self-consistency) | 0.224 | 0.089 | 0.164 |
| Verbalized (est. of 100 annotators) | **0.072** | **0.863** | **0.393** (p ≈ 1e-113) |
| Human pool | — | 0.939 | — |

(Full set, n = 3,113.) Verbalized distributions are ~3× closer to the human
distribution than sampled ones, their average entropy nearly matches the human
pool's, and verbalized entropy clears the pre-registered routing threshold
(ρ = 0.393 ≥ 0.3; pilot estimate was 0.354). Where sampled JSD degrades 4.2×
from the lowest- to highest-entropy quintile (§4.1), verbalized JSD is nearly
flat: 0.044 → 0.103. The declarative estimate stays accurate on precisely the
contested items where the behavioral distribution collapses. Strikingly, the
two uncertainty signals are essentially uncorrelated with each other
(ρ = 0.03–0.08, depending on which sampled run is paired): they measure
different quantities. Together with
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

### 4.5 The router: uncertainty finds hard items, not model errors

The simulation (method in §3) asks the operational question directly: given a
fixed annotation budget, does routing on model-reported uncertainty beat
spending the same budget at random? The answer splits cleanly by what the
unrouted items fall back to.

**Against the sampled base** — the realistic setting where un-routed items
keep the model's self-consistency distribution — verbalized-entropy routing
works (figure: `figures/router_sampled_examples.png`). At k = 10 annotations
per routed item it captures **37.4%** of the expected-gain oracle's
improvement over random; channel disagreement (JSD between the model's two
distributions, using no human information) captures **45.5%**, statistically
indistinguishable from the *true-human-entropy* oracle (47.7%). In budget
terms, random allocation needs **1.44–1.57×** as many annotations to match
verbalized-entropy routing at 10–30% coverage (strategy–random gap CIs
exclude zero at every budget above 5%). Self-consistency entropy captures
nothing (−8.3%), closing the loop on §4.2 with an end-to-end null.

**Against the verbalized base** — where un-routed items keep the model's
verbalized distribution, the stronger baseline §4.3 establishes — the same
signal *inverts* (figure: `figures/router_verb_examples.png`): verbalized
entropy is reliably worse than random (capture −18.7%; random matches it with
0.5–0.6× the budget). The mechanism is anti-selection. High verbalized
entropy flags items where humans genuinely disagree — but §4.3 showed the
verbalized distribution is already accurate exactly there. The residual
errors sit in low-entropy items the model is confidently wrong about, which
entropy routing systematically deprioritizes (per-item verbalized entropy vs.
verbalized error: ρ ≈ −0.07). Even the true-human-entropy oracle barely beats
random here (11.9%), while the expected-gain oracle still improves
substantially — the errors are findable in principle, just not by any
entropy-shaped signal. **Model-reported uncertainty tells you where the task
is hard, not where the model is wrong**; it routes usefully only when the
fallback annotator is weak.

**Breadth vs. depth.** Sweeping k (figures:
`figures/router_ksweep_{sampled,verb}_examples.png`) shows that at a fixed
total budget, shallow-and-broad beats deep-and-narrow on the sampled base
(k = 3–5 dominates k = 20–50), and yields the study's most quotable number on
the verbalized base: replacing the model's verbalized distribution with the
empirical distribution of k human labels only breaks even at **k ≈ 5**
(k = 1: JSD 0.270; k = 3: 0.113; k = 5: 0.072; model verbalized: 0.076). In
distributional terms, one model call is worth roughly five human annotations
on this dataset — with the memorization caveat of §4.3 attached.

All orderings and magnitudes are stable under total variation distance in
place of JSD and under the bootstrap split protocol (capture 37.7–38.9%
verbalized, 44.0–46.8% channel disagreement on the sampled base;
anti-selection persists on the verbalized base in every variant).

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
- **The router is a simulation, not a deployment.** Routed items draw
  annotations from ChaosNLI's own pool, so annotators are exchangeable with
  the evaluation target by construction; a real deployment adds annotator
  drift, and its ground truth would not be a held-out half of 100 labels. The
  memorization caveat (§4.3) also propagates: if verbalized distributions are
  partly recalled, both the verbalized base and the routing signal are
  optimistic. The sampled-base conclusions are less exposed, since the
  sampled distributions show no memorization signature (§4.4).
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

For the substitution debate, the results assemble into one architecture with
a warning label. The stratified curve (§4.1) says the cheap examples are the
ones the model already labels well — consistent with aggressive automation of
the easy majority. The failed gate (§4.2) says the obvious triage signal —
"let the model flag what it's unsure about," read off its sampling behavior —
does not exist. The verbalized result (§4.3) supplies the repair: the signal
is a *declarative* capability that must be asked for, and the router built on
it (§4.5) delivers real budget savings (~1.5× annotation efficiency) over
random allocation in the realistic setting. The warning label is the
anti-selection result: the same uncertainty signal is worse than useless for
auditing the model's own best output, because model-reported uncertainty
locates task difficulty, not model error. Practically: use verbalized
distributions as the machine annotation, route by uncertainty only to decide
*which items get humans at all*, and do not expect the model to tell you
where it is wrong — error-finding needs an independent signal. Regardless,
headline agreement numbers should be reported stratified by inter-annotator
agreement as a matter of course.

## Reproducibility

`scripts/prepare.py` → `scripts/elicit.py submit|status|collect` →
`scripts/analyze.py`; verbalized arm: `scripts/verbalize.py`; router
simulation: `scripts/router.py` (pure simulation on saved elicitation
outputs; no further API calls). Elicitation: claude-sonnet-5, Message Batches
API (verbalized full set ran synchronously), 10 samples/item, randomized
option order, JSON-schema output, thinking disabled, seed 20260805 for v0–v1
randomization and bootstraps, seed 20260819 for the router simulation. Total
API cost: ≈ $20–27 (34,243 requests; ~12.5M input / ~0.4M output tokens).
