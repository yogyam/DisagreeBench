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
(figure: `figures/jsd_vs_entropy_examples.png`). This deliberately replicates
the stratified diagnostic Nie et al. (2020) ran on fine-tuned encoders, and
the shape has changed in a way that matters: where their models' accuracy
fell to chance on contested items, the frontier LLM stays well above it, and
its Q1 JSD now reaches their estimated human bound (~0.06). Accuracy against
the majority declines much more slowly, and its floor is deceptive: on Q5 items the human
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
enabled (§6) leaves the picture unchanged. By the pre-registered logic of the
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

The direction of this contrast — verbalized beats sampled, and reasoning does
not close the gap — independently replicates what Ni et al. (2026) report on
hate-speech, emotion, and preference tasks and Meister et al. (2025) on
opinion surveys; §5.3 details what is new here (the entropy-flatness, the
channel independence, and the routing consequences in §4.5).

One caveat attaches specifically to this result: ChaosNLI's label
*distributions* have been public since 2020 — and are by now an explicit
optimization target in the literature (e.g., SHALA-LLM fine-tunes directly on
them) — so the model may have absorbed some of them in pretraining. A
memorized distribution would inflate verbalized performance in a way the
gold-label diagnostic in §4.4 does not cover. Replication on held-out or
post-cutoff annotation data is required before the verbalized result is
load-bearing.

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

## 5. Related Work

### 5.1 Human label variation and distributional evaluation

That disagreement in NLI judgments is reproducible signal rather than
annotation noise was established by Pavlick and Kwiatkowski (2019), and the
position that label variation should be modeled rather than adjudicated away
is now a program (Plank, 2022; Uma et al., 2021; the perspectivist manifesto,
Basile et al., 2021, published as Cabitza et al., 2023). Our evaluation
toolkit is theirs: JSD against the human label distribution and the
correlation between per-item model and crowd entropy are both standardized in
Uma et al. (2021). Our dataset paper, ChaosNLI (Nie et al., 2020), already
performed the stratified analysis for fine-tuned encoders: accuracy degrades
toward chance and JSD stays above 0.2 as human agreement falls. §4.1 is a
deliberate replication of that diagnostic on a frontier LLM, and what changed
is the shape: the model now reaches the estimated human JSD bound (~0.06) on
high-agreement items, and accuracy on contested items no longer collapses to
chance (0.56 vs. 0.33) — which makes the flattery problem *worse*, because
the standard metric now looks respectable everywhere while distributional
divergence quadruples. Distributed NLI (Zhou et al., 2022) formalized the
task of predicting human opinion distributions on ChaosNLI: their supervised
estimators (MC Dropout, recalibration, distillation) reach JSD ≈ 0.18–0.19,
which our zero-supervision verbalized elicitation beats by ~2.5×; their
appendix also documents that encoder uncertainty is miscalibrated against
human disagreement, a precursor of our gate analysis. Baan et al. (2022)
showed on ChaosNLI that calibration to a majority label is incoherent under
disagreement (our 50/50 truth/pool split in §3 adapts their device of
splitting the 100 annotations into sub-populations); Baan et al. (2024) argue
conceptually that a predictive distribution conflates model confidence with
human variation — our channel-decorrelation finding (§4.3) turns that
distinction into a measured property. VariErr NLI (Weber-Genzel et al., 2024)
implies some of ChaosNLI's variation is annotation error, which makes our
router's measured gains conservative with respect to error-cleaned targets.

### 5.2 LLMs as annotators

The replacement wave (Gilardi et al., 2023; Törnberg, 2023) reads the model's
high self-consistency as annotation quality; on contested items we show the
same property is a liability — the model agrees with itself (α = 0.90) about
questions on which 100 humans split nearly evenly (α = 0.37). Reiss (2023)
complained that 2023-era ChatGPT was too *unstable* to annotate reliably;
2026 frontier models exhibit the opposite pathology, and neither regime
tracks human disagreement. Baumann et al. (2025) quantify the downstream
stakes — LLM annotation choices flip roughly a third of tested conclusions —
and already propose verbalized-confidence-targeted human annotation as a
mitigation, evaluated against disagreement-filtered gold labels; our
anti-selection result (§4.5) exposes a failure mode of exactly that strategy
once the target is the label distribution and the fallback is the model's
own best output.

### 5.3 LLM uncertainty channels and human disagreement

Lee et al. (2023) first compared sampled and logprob-derived LLM
distributions to ChaosNLI's annotator distributions, observing that GPT-3's
output entropy is near zero and visibly uncorrelated with human entropy. Our
§4.2 quantifies that observation at frontier scale (85% exact determinism;
α = 0.90 vs. 0.37; gate ρ = 0.164 with CIs) and stress-tests it against
reasoning and paraphrase confounds. Madaan et al. (2025) stratify accuracy by
human entropy for open-weight models on ChaosNLI; the accuracy/JSD
*dissociation* across quintiles is ours. The channel contrast we find was
anticipated on other tasks: Ni et al. (2026) show verbalized distributions
beat sampled ones for disagreement prediction on hate-speech, emotion, and
preference tasks (and that RLVR-style reasoning hurts); Meister et al. (2025)
found LLMs describe opinion distributions better than they simulate them;
Jang et al. (2026) attribute the simulation failure to alignment-induced mode
collapse (cf. Kirk et al., 2024), consistent with the calibration literature
where asking beats token probabilities (Lin et al., 2022; Tian et al., 2023;
Kadavath et al., 2022) — though against *correctness*, consistency-based
signals often win (Xiong et al., 2024), underlining that our target is
different. Wang et al. (2024) show first-token probabilities diverge from
what instruction-tuned models actually answer, which is why we omit a
logprob arm. On NLI specifically, Chen et al. (2024; 2025) approximate human
judgment distributions with explanation-conditioned LLM prompting, and Chen
et al. (2026) show CoT does not calibrate distributions on ambiguous items.
Relative to all of these, our §4.3–4.4 contribute: replication of the
verbalized–sampled gap on 100-way NLI distributions with a frontier model;
the near-flatness of verbalized JSD across disagreement levels; the
statistical *independence* of the two channels' entropies (ρ ≤ 0.08), which
no prior work reports; and the gate framing that connects channel quality to
routability. Zhang et al. (2025) propose verbalized sampling to recover
output diversity; our results externally validate the verbal channel against
ground-truth human distributions rather than diversity alone.

### 5.4 Hybrid annotation and budget allocation

Routing between model and human annotators instantiates learning-to-defer
(Madras et al., 2018; Mozannar & Sontag, 2020; DeSalvo et al., 2025), but
that literature assumes a trained rejector and single-label accuracy; we ask
which zero-shot LLM signal could support deferral at all, with distributional
fidelity as the objective. CoAnnotating (Li et al., 2023) routes on
prompt-paraphrase entropy against single gold labels and reports scalar
verbalized confidence unreliable — the opposite ordering from ours, which
dissolves once the elicitation target is distinguished: they elicit scalar
self-confidence scored against gold labels; we elicit a full annotator
distribution scored against real human distributions. HyPAC (Zeng et al.,
2026) gives PAC-guaranteed routing for 0-1 error against objective answers;
production systems (Kim et al., 2024; Bachar et al., 2026) route escalation
on uncertainty and find raw signals wanting. Gligorić et al. (2025) allocate
human budget by verbalized confidence for population-level estimates;
Mehrotra et al. (2026) do so at demographic-group level; Hakimi et al.
(2026) find uncertainty-driven selection fails to beat random in active
learning; Schroeder et al. (2025) show humans anchor when *verifying* LLM
labels, which motivates our design choice of independent annotation on
routed items. Closest to our simulation: Klugmann et al. (2024) route items
between crowd-trained soft-label predictors and humans in vision, and Peale
et al. (2026) run budget-swept uncertainty-decomposition routing on ChaosNLI
with trained classifiers. Kohli (2026) shows on ChaosNLI that annotations
needed per item is metric-dependent (distributional metrics saturate near
N = 10), consistent with our k-sweep; Gruber et al. (2025) pose the
annotator-selection question — whether a human or a model provides each
label — as an open empirical problem. No prior work runs the experiment in
§4.5: an item-level budget sweep over both coverage and annotation depth
against real 100-way label distributions, with regret oracles, comparing
elicitation channels as routing signals *and* as fallbacks. The
fallback-channel reversal — the same uncertainty signal beats random against
a weak fallback and anti-selects against a strong one — appears in none of
the above.

## 6. Limitations

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

## 7. Implications

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

## References

*(Draft list, compiled and URL-verified 2026-08-19; formatting to be normalized at submission. Exact titles/venues for Schroeder et al. and any 2026 arXiv-only entries should be re-checked at camera-ready.)*

- Baan, Joris, Raquel Fernández, Barbara Plank, and Wilker Aziz (2024). Interpreting Predictive Probabilities: Model Confidence or Human Label Variation? EACL 2024 (Volume 2: Short Papers), pp. 268-277. [Verified; it is a position paper at EACL 2024 main conference, matching the description.] https://aclanthology.org/2024.eacl-short.24/
- Basile, V., Cabitza, F., Campagner, A., & Fell, M. (2021). Toward a Perspectivist Turn in Ground Truthing for Predictive Computing. arXiv:2109.04270. NOTE: the peer-reviewed version is Cabitza, F., Campagner, A., & Basile, V. (2023), AAAI-23, DOI 10.1609/aaai.v37i6.25840 — different author order and year. Citing 'Basile et al. (2021)' is only correct for the arXiv preprint; cite Cabitza et al. (2023) for the published version. https://arxiv.org/abs/2109.04270
- Basile, V., Fell, M., Fornaciari, T., Hovy, D., Paun, S., Plank, B., Poesio, M., & Uma, A. (2021). We Need to Consider Disagreement in Evaluation. Proceedings of the 1st Workshop on Benchmarking: Past, Present and Future (ACL 2021), pp. 15-21. [Included because 'Basile et al. (2021) perspectivist manifesto' is ambiguous — this is the other paper that citation string commonly resolves to, and the one actually published in an NLP venue in 2021.] https://aclanthology.org/2021.bppf-1.3/
- Baumann, Röttger, Urman, Wendsjö, Plaza-del-Arco, Gruber & Hovy (2025). Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation. arXiv:2509.08825 https://arxiv.org/abs/2509.08825
- Chaemin Jang, Dongman Lee, and Jihee Kim (2026). Instruction-Tuned Language Models Cannot Sample from Distributions They Can Describe. arXiv preprint arXiv:2607.25292. https://arxiv.org/abs/2607.25292
- Chen, B., Hu, T., Zhang, C., Litschko, R., Korhonen, A., & Plank, B. (2026). Decoupling the Effect of Chain-of-Thought Reasoning: A Human Label Variation Perspective. Findings of ACL 2026. arXiv:2601.03154. https://arxiv.org/abs/2601.03154
- Chen, B., Peng, S., Korhonen, A., & Plank, B. (2025). A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI. Findings of ACL 2025. https://arxiv.org/abs/2412.13942
- Chen, B., Wang, X., Peng, S., Litschko, R., Korhonen, A., & Plank, B. (2024). "Seeing the Big through the Small": Can LLMs Approximate Human Judgment Distributions on NLI from a Few Explanations? Findings of EMNLP 2024. arXiv:2406.17600 https://arxiv.org/abs/2406.17600
- DeSalvo, G., Mohri, C., Mohri, M., & Zhong, Y. (2025). Budgeted Multiple-Expert Deferral. arXiv:2510.26706 (preprint, submitted October 2025; no peer-reviewed venue listed). https://arxiv.org/abs/2510.26706
- Ellie Pavlick and Tom Kwiatkowski (2019). Inherent Disagreements in Human Textual Inferences. Transactions of the Association for Computational Linguistics, vol. 7, pp. 677–694. https://aclanthology.org/Q19-1043/
- Gilardi, Alizadeh & Kubli (2023). ChatGPT outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy of Sciences 120(30), e2305016120 https://www.pnas.org/doi/10.1073/pnas.2305016120
- Gligorić, K., Zrnic, T., Lee, C., Candès, E., & Jurafsky, D. (2025). Can Unconfident LLM Annotations Be Used for Confident Conclusions? Proceedings of NAACL 2025 (Volume 1: Long Papers), pages 3514–3533, Albuquerque, New Mexico. ACL. https://aclanthology.org/2025.naacl-long.179/
- Gruber, Cornelia, Helen Alber, Bernd Bischl, Göran Kauermann, Barbara Plank, and Matthias Aßenmacher (2025). Revisiting Active Learning under (Human) Label Variation. Proceedings of the 4th Workshop on Perspectivist Approaches to NLP (NLPerspectives 2025). NOTE: the task description overstates this paper — it does NOT explicitly call for LLM-assisted annotation-budget-allocation experiments as a combined agenda; it discusses LLM annotators and budgets separately and calls generally for empirical validation. https://aclanthology.org/2025.nlperspectives-1.7/
- Hakimi, A. D., Hirlimann, L., Augenstein, I., & Schütze, H. (2026). Do We Still Need Humans in the Loop? Comparing Human and LLM Annotation in Active Learning for Hostility Detection. arXiv:2604.13899 https://arxiv.org/abs/2604.13899
- Hannah Kim, Kushan Mitra, Rafael Li Chen, Sajjadur Rahman, Dan Zhang (2024). MEGAnno+: A Human-LLM Collaborative Annotation System. EACL 2024 System Demonstrations. https://aclanthology.org/2024.eacl-demo.18/
- Hao Zeng, Huipeng Huang, Xinhao Qu, Jianguo Huang, Bingyi Jing, Hongxin Wei (2026). HyPAC: Cost-Efficient LLMs-Human Hybrid Annotation with PAC Error Guarantees. arXiv:2602.02550 (preprint; listed as under consideration at JASA, not yet a confirmed venue). https://arxiv.org/abs/2602.02550
- Joris Baan, Wilker Aziz, Barbara Plank, and Raquel Fernández (2022). Stop Measuring Calibration When Humans Disagree. Proceedings of EMNLP 2022, pp. 1892–1915, Abu Dhabi. https://aclanthology.org/2022.emnlp-main.124/
- Kadavath, S., Conerly, T., Askell, A., Henighan, T., et al. (2022). Language Models (Mostly) Know What They Know. arXiv preprint arXiv:2207.05221 (Anthropic; not peer-reviewed venue) https://arxiv.org/abs/2207.05221
- Kirk, R., Mediratta, I., Nalmpantis, C., Luketina, J., Hambro, E., Grefenstette, E., & Raileanu, R. (2024). Understanding the Effects of RLHF on LLM Generalisation and Diversity. ICLR 2024. arXiv:2310.06452 https://arxiv.org/abs/2310.06452
- Klugmann, Christopher, Rafid Mahmood, Guruprasad Hegde, Amit Kale, and Daniel Kondermann (2024). No Need to Sacrifice Data Quality for Quantity: Crowd-Informed Machine Annotation for Cost-Effective Understanding of Visual Data. arXiv preprint arXiv:2409.00048. NOTE: no peer-reviewed venue found; cite as arXiv preprint. https://arxiv.org/abs/2409.00048
- Kohli, Guneet (2026). Metric-Dependent Annotation Saturation for Learning from Label Distributions. arXiv preprint arXiv:2605.29797. NOTE: single-author paper — cite as Kohli (2026), not 'Kohli et al.'; arXiv preprint, no peer-reviewed venue found. https://arxiv.org/abs/2605.29797
- Lee, N., An, N. M., & Thorne, J. (2023). Can Large Language Models Capture Dissenting Human Voices? EMNLP 2023. arXiv:2305.13788. https://arxiv.org/abs/2305.13788
- Lin, S., Hilton, J., & Evans, O. (2022). Teaching Models to Express Their Uncertainty in Words. Transactions on Machine Learning Research (TMLR). arXiv:2205.14334 https://arxiv.org/abs/2205.14334
- Lovish Madaan, David Esiobu, Pontus Stenetorp, Barbara Plank, and Dieuwke Hupkes (2025). Lost in Inference: Rediscovering the Role of Natural Language Inference for Large Language Models. Proceedings of NAACL 2025 (Volume 1: Long Papers), pages 9229–9242, Albuquerque. ACL. (arXiv:2411.14103, Nov 2024 — the task's 'Madaan et al. 2024' refers to the preprint; the published venue is NAACL 2025.) https://aclanthology.org/2025.naacl-long.466/
- Madras, D., Pitassi, T., & Zemel, R. (2018). Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer. Advances in Neural Information Processing Systems 31 (NeurIPS 2018). (arXiv:1711.06664; earlier workshop version titled 'Increasing Fairness by Learning to Defer') https://arxiv.org/abs/1711.06664
- Mehrotra, N., Visokay, A., & Gligorić, K. (2026). Multi-Perspective LLM Annotations for Valid Analyses in Subjective Tasks. arXiv:2603.21404 (Mar 22, 2026) https://arxiv.org/abs/2603.21404
- Minzhi Li, Taiwei Shi, Caleb Ziems, Min-Yen Kan, Nancy F. Chen, Zhengyuan Liu, Diyi Yang (2023). CoAnnotating: Uncertainty-Guided Work Allocation between Human and Large Language Models for Data Annotation. EMNLP 2023 (main, Singapore). https://aclanthology.org/2023.emnlp-main.92/
- Mozannar, H., & Sontag, D. (2020). Consistent Estimators for Learning to Defer to an Expert. Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR 119:7076–7087. https://proceedings.mlr.press/v119/mozannar20b.html
- Ni, Fan, Zouhar, Rooein, Hoyle, Sachan, Leippold, Hovy & Ash (2026). Can Reasoning Help Large Language Models Capture Human Annotator Disagreement? EACL 2026 Main (arXiv:2506.19467; v1 Jun 2025, v3 Jan 2026). Note: exact title differs slightly from the working description — it foregrounds reasoning, not verbalized-vs-sampling. https://arxiv.org/abs/2506.19467
- Nicole Meister, Carlos Guestrin, and Tatsunori Hashimoto (2025). Benchmarking Distributional Alignment of Large Language Models. Proceedings of NAACL-HLT 2025 (Volume 1: Long Papers), pages 24–49, Albuquerque, New Mexico. Association for Computational Linguistics. (arXiv:2411.05403) https://aclanthology.org/2025.naacl-long.2/
- Or Bachar, Or Levi, Sardhendu Mishra, Adi Levi, Manpreet Singh Minhas, Justin Miller, Omer Ben-Porat, Eilon Sheetrit, Jonathan Morra (2026). LLM Performance Predictors: Learning When to Escalate in Hybrid Human-AI Moderation Systems. AAMAS 2026 (arXiv:2601.07006; Zefr). https://arxiv.org/abs/2601.07006
- Peale, C., Devic, S., Gopalan, P., Wieder, U., & Gollakota, A. (2026). Flexible Routing via Uncertainty Decomposition. arXiv:2605.07805 https://arxiv.org/abs/2605.07805
- Plank, B. (2022). The "Problem" of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation. Proceedings of EMNLP 2022, pp. 10671-10682, Abu Dhabi. https://aclanthology.org/2022.emnlp-main.731/
- Reiss (2023). Testing the Reliability of ChatGPT for Text Annotation and Classification: A Cautionary Remark. arXiv:2304.11085 (preprint; no journal venue found) https://arxiv.org/abs/2304.11085
- Schroeder, H., Roy, D., & Kabbara, J. (2025). Human-LLM Interactions Reveal Anchoring Effects in Annotation. Findings of ACL 2025. (anchoring in human review of LLM label suggestions)
- Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C. D. (2023). Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. EMNLP 2023 https://aclanthology.org/2023.emnlp-main.330/
- Törnberg (2023). ChatGPT-4 Outperforms Experts and Crowd Workers in Annotating Political Twitter Messages with Zero-Shot Learning. arXiv:2304.06588 [journal version: Törnberg (2025), Large Language Models Outperform Expert Coders and Supervised Classifiers at Annotating Political Social Media Messages, Social Science Computer Review 43(6), 1181-1195] https://arxiv.org/abs/2304.06588
- Uma, A. N., Fornaciari, T., Hovy, D., Paun, S., Plank, B., & Poesio, M. (2021). Learning from Disagreement: A Survey. Journal of Artificial Intelligence Research (JAIR), 72, 1385-1470. https://www.jair.org/index.php/jair/article/view/12752
- Wang, X., Ma, B., Hu, C., Weber-Genzel, L., Röttger, P., Kreuter, F., Hovy, D., & Plank, B. (2024). "My Answer is C": First-Token Probabilities Do Not Match Text Answers in Instruction-Tuned Language Models. Findings of ACL 2024 https://aclanthology.org/2024.findings-acl.441/
- Weber-Genzel, Leon, Siyao Peng, Marie-Catherine de Marneffe, and Barbara Plank (2024). VariErr NLI: Separating Annotation Error from Human Label Variation. ACL 2024 (Volume 1: Long Papers), Bangkok. [Verified; one detail in our prompt was off: the data is 500 re-annotated MNLI items, not ChaosNLI — MNLI overlaps ChaosNLI's source but VariErr's annotations are their own 2-round procedure.] https://aclanthology.org/2024.acl-long.123/
- Wu, J., Wang, A., Ong, K., Liang, P. P., & Picard, R. (2026). SHALA-LLM: Smartly Handling Ambiguous Labels in Aligning LLMs. arXiv:2606.05376 (Jun 3, 2026) https://arxiv.org/abs/2606.05376
- Xiang Zhou, Yixin Nie, Mohit Bansal (2022). Distributed NLI: Learning to Predict Human Opinion Distributions for Language Reasoning. Findings of the Association for Computational Linguistics: ACL 2022. https://aclanthology.org/2022.findings-acl.79/
- Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2024). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. ICLR 2024. arXiv:2306.13063 https://arxiv.org/abs/2306.13063
- Yixin Nie, Xiang Zhou, Mohit Bansal (2020). What Can We Learn from Collective Human Opinions on Natural Language Inference Data? Proceedings of EMNLP 2020, pages 9131-9143. https://aclanthology.org/2020.emnlp-main.734/
- Zhang, J., Yu, S., Chong, D., Sicilia, A., Tomz, M. R., Manning, C. D., & Shi, W. (2025). Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity. arXiv preprint arXiv:2510.01171 https://arxiv.org/abs/2510.01171

## Reproducibility

`scripts/prepare.py` → `scripts/elicit.py submit|status|collect` →
`scripts/analyze.py`; verbalized arm: `scripts/verbalize.py`; router
simulation: `scripts/router.py` (pure simulation on saved elicitation
outputs; no further API calls). Elicitation: claude-sonnet-5, Message Batches
API (verbalized full set ran synchronously), 10 samples/item, randomized
option order, JSON-schema output, thinking disabled, seed 20260805 for v0–v1
randomization and bootstraps, seed 20260819 for the router simulation. Total
API cost: ≈ $20–27 (34,243 requests; ~12.5M input / ~0.4M output tokens).
