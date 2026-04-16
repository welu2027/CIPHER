# CIPHER
### Calibrated Introspection via Partially Hidden Environment Rules

CIPHER is a procedurally-generated benchmark designed to test whether language models actually know what they know — and what they don't. Every instance is a tiny invented world with its own causal rules, but some of those rules are deliberately hidden. The model has to figure out how much it can trust its own understanding, rank which gaps matter most, probe the system if it wants, commit to a plan, and then honestly assess how robust that plan is.

The whole point is that no model can memorize its way through this. Every world uses made-up vocabulary — invented entity names, invented property words, invented causal language — generated fresh from abstract math. If a model scores well, it's because it genuinely reasoned under uncertainty, not because it pattern-matched on something from training.

## What's in this dataset

```
cipher/
  __init__.py
  world.py          state representation, rules, action engine
  generator.py      procedural instance generator (seeded, fully deterministic)
  simulator.py      executes a model's plan against the hidden rules
  scorer.py         computes all four scoring dimensions
  schema.py         validates and parses model JSON output
  prompt.py         builds the natural-language prompt for each instance
  flavor.py         procedural vocabulary layer (invented terms per instance)
  optimal.py        beam-search oracle for computing normalized scores
data/
  instances.jsonl   1,000 pre-generated instances (seed=2026, with oracle bounds)
scripts/
  generate_dataset.py   regenerate the benchmark at any seed/size
```

The `data/instances.jsonl` file has everything needed to run evaluations without regenerating. Each line is one instance with a `prompt` field (what the model sees) and a `hidden` field (ground truth used for scoring — not shown to the model).

## How scoring works

Each model response is scored on four dimensions, all normalized to [0, 1]:

| Dimension | Weight | What it's measuring |
|-----------|--------|---------------------|
| **Objective** | 35% | How good is the final plan vs. the oracle beam search? |
| **Calibration** | 25% | Brier score on the model's stated confidence in its own knowledge |
| **Attention** | 20% | Does the model rank the important unknowns above the unimportant ones? |
| **Executive** | 20% | Plan structure: named risks, alternative plans, probe strategy |

The composite is a weighted average. One thing worth noting: no simple strategy wins all four dimensions at once. A model that always plans greedily gets a great objective score but zero attention and poor calibration. A model that hedges everywhere gets decent calibration but a bad objective. A model that genuinely reasons about what it doesn't know — and acts accordingly — is the one that scores well across the board.

## Baseline scores (1,000 instances, seed=2026)

| agent | composite | objective | calibration | attention | executive |
|-------|-----------|-----------|-------------|-----------|-----------|
| stub-noop | 0.408 | 0.486 | 0.750 | 0.000 | 0.250 |
| stub-random | 0.511 | 0.484 | 0.663 | 0.211 | 0.670 |
| stub-greedy | 0.623 | 1.000 | 0.893 | 0.000 | 0.250 |

These are floor/ceiling references, not targets. The greedy stub scores 1.0 on objective because it runs beam search on the visible rules — but it claims everything is known with high confidence and never identifies the unknowns that actually matter, so its calibration and attention are poor. Real models should do meaningfully better on the composite.

## Regenerating the dataset

The included `data/instances.jsonl` is ready to use, but if you want to regenerate it at a different seed or size:

```bash
python3 scripts/generate_dataset.py --n 1000 --out data/instances.jsonl --seed 2026 --oracle
```

The `--oracle` flag pre-computes the best and worst achievable objectives for each instance (used to normalize scores). It adds ~40s for 1,000 instances.

## Kaggle Benchmark

This dataset is the backing store for the CIPHER Kaggle Benchmark, which evaluates frontier LLMs — Gemini, Claude, GPT-4o, and open-source models — against these instances. The benchmark notebook attaches this dataset, loads `instances.jsonl`, and runs each model through `cipher_task` via the `kaggle-benchmarks` SDK.

The benchmark is part of the Measuring Progress Toward AGI hackathon, targeting the **Metacognition** track.
