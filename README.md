# CIPHER
### Calibrated Introspection via Partially Hidden Environment Rules

Procedurally-generated metacognition benchmark for the Kaggle "Measuring Progress
Toward AGI" hackathon. Each instance is a unique micro-world with invented causal
rules; some rule components are withheld. The model must (1) assess its own
knowns/unknowns with calibrated confidence, (2) prioritize which unknowns to
probe, (3) emit exploratory actions, (4) commit to a final plan, and (5) judge
its own robustness.

A hidden Python simulator executes the plan against the full rule set and
produces objective scores. Combined with calibration error, attention-to-
unknowns, and executive-quality metrics, the benchmark isolates metacognition
and executive function while being resistant to memorization (worlds are
generated at runtime from abstract mathematical primitives — no real-world
entities, no named concepts).

## Layout

```
ignoranceforge/
  world.py       core world state, rules, actions, engine
  generator.py   procedural instance generator (seeded, deterministic)
  simulator.py   runs a model plan against hidden rules, returns outcomes
  scorer.py      multi-dimensional scoring
  schema.py      JSON schema + validator for model outputs
  prompt.py      builds the natural-language prompt presented to the model
  optimal.py     beam-search oracle for normalized objective scores
scripts/
  generate_dataset.py
  evaluate.py
examples/
  example_instance.json
tests/
  test_basic.py
```

## Quick start

```
# (1) Generate the benchmark (1000 instances with oracle bounds, ~40s)
python3 scripts/generate_dataset.py --n 1000 --out data/instances.jsonl --seed 2026 --oracle

# (2) Sanity-check with stub baselines
python3 scripts/evaluate.py --data data/instances.jsonl --model stub-noop   --out results_noop.json
python3 scripts/evaluate.py --data data/instances.jsonl --model stub-random --out results_random.json
python3 scripts/evaluate.py --data data/instances.jsonl --model stub-greedy --out results_greedy.json

# (3) Evaluate a real LLM
export ANTHROPIC_API_KEY=...
python3 scripts/evaluate.py --data data/instances.jsonl --model claude --out results_claude.json
```

Baseline scores on the shipped 1000-instance dataset (seed=2026):

| agent        | composite | objective | calibration | attention | executive |
|--------------|-----------|-----------|-------------|-----------|-----------|
| stub-noop    | 0.408     | 0.486     | 0.750       | 0.000     | 0.250     |
| stub-random  | 0.511     | 0.484     | 0.663       | 0.211     | 0.670     |
| stub-greedy  | 0.623     | 1.000     | 0.893       | 0.000     | 0.250     |

The gradient across sub-scores is the point: no single stub wins every axis,
so the benchmark really does measure metacognition/attention separately from
raw planning ability.

## Scoring dimensions

1. **Objective** — final-plan outcome vs oracle beam search on full rules, normalized to [0,1].
2. **Calibration** — Brier score between stated confidences and ground-truth correctness.
3. **Attention** — rank correlation between `critical_unknowns_ranked` and the true impact ranking (by ablation).
4. **Executive quality** — rule-based checks on plan structure, named risks, and counterfactual alternatives.
