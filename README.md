# DisagreeBench

Where LLM annotation substitutes for human labelers — and where it doesn't.
Design doc: see the project draft. Current scope: **v0 (stratified agreement
study) + v1 (the gate)** on ChaosNLI (SNLI + MNLI), one model.

**Method:** self-consistency sampling (10 samples/example) with `claude-sonnet-5`
via the Message Batches API (50% price). Thinking disabled, output constrained
to a JSON label schema, label option order randomized per sample.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # or `ant auth login`
```

## Pipeline

```sh
# 1. Data prep — entropy quintiles + stratified 200-example pilot
.venv/bin/python scripts/prepare.py

# 2. Pilot elicitation (~2,000 batch requests, ≈ $1)
.venv/bin/python scripts/elicit.py submit --split pilot
.venv/bin/python scripts/elicit.py status --tag pilot     # poll until "ended"
.venv/bin/python scripts/elicit.py collect --tag pilot

# 3. Analysis — JSD-vs-entropy curve, the Spearman gate, contamination check
.venv/bin/python scripts/analyze.py --tag pilot --split pilot

# Full run (~45,000 requests, ≈ $11–15 batched) once the pilot validates:
.venv/bin/python scripts/elicit.py submit --split examples
.venv/bin/python scripts/elicit.py collect --tag examples
.venv/bin/python scripts/analyze.py --tag examples --split examples
```

## Layout

```
data/raw/        ChaosNLI v1.0 (downloaded)
data/processed/  examples.jsonl (unified, bucketed), pilot.jsonl
results/         batch metadata, raw samples, per-example model distributions
figures/         jsd_vs_entropy, gate_scatter, contamination
scripts/         prepare.py, elicit.py, analyze.py
```

## Notes / deviations from the design doc

- Sampling temperature is not tunable on claude-sonnet-5 (the API rejects
  non-default sampling params), so the temperature-sensitivity sweep in §9 is
  moot; sampling runs at the model default.
- Thinking is disabled to keep cost per sample minimal; note this as a
  limitation (a reasoning pass could shift the label distribution).
- Contamination diagnostic (§9) is built in: `analyze.py` compares agreement
  with the original gold label vs. the ChaosNLI majority, per entropy bucket.
