# CIPHER
### Calibrated Introspection via Partially Hidden Environment Rules

CIPHER is a procedurally generated benchmark for planning under missing information. Each instance is a small invented world with causal rules, some of which are withheld from the prompt. The model reports what it knows, ranks the hidden rules by importance, optionally declares probes, commits to a plan, and gives a contingency plan.

Model evaluations were run on Kaggle Benchmarks under a Kaggle grant.

## Layout

```
cipher/        core library (no runtime dependencies)
  scorer.py      scorer as submitted (v1)
  scorer_v2.py   label normalised calibration and attention (v2)
data/          instances.jsonl: 1,000 instances, seed 2026, with oracle bounds
scripts/       generate_dataset.py, evaluate.py (stub baselines)
notebooks/     Kaggle Benchmarks notebooks
analysis/      re-analysis scripts; outputs in analysis/out/ (see its README)
results/       model and baseline scores, reports, figures
```

## Scoring

| Dimension | Weight | Measures |
|---|---|---|
| Objective | 35% | Final plan vs. beam search oracle |
| Calibration | 25% | Brier score on stated knowledge of each rule component |
| Attention | 20% | Ranking of hidden rules vs. true impact |
| Executive | 20% | Probe declarations and contingency plan |

Calibration in v1 matches claims to internal rule names the prompt never shows, so it mostly measures label formatting. v2 fixes this; see `analysis/out/README.md`.

## Baselines (1,000 instances)

| Agent | Composite | Objective | Calibration | Attention | Executive |
|---|---|---|---|---|---|
| `stub-noop` | 0.358 | 0.486 | 0.750 | 0.000 | 0.000 |
| `stub-greedy` | 0.473 | 0.865 | 0.680 | 0.000 | 0.000 |
| `stub-random` | 0.521 | 0.478 | 0.669 | 0.532 | 0.400 |
| `stub-cautious` | 0.681 | 0.481 | 0.990 | 0.676 | 0.648 |
| `stub-probe-heavy` | 0.726 | 0.761 | 0.897 | 0.676 | 0.500 |

## Usage

```bash
python3 scripts/evaluate.py --model stub-greedy                     # one baseline
python3 scripts/generate_dataset.py --n 1000 --seed 2026 --oracle   # regenerate data
.venv/bin/python analysis/run_reanalysis.py                         # full re-analysis
```

The re-analysis needs `requirements-analysis.txt` and makes no API calls.
