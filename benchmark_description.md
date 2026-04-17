## CIPHER: Calibrated Introspection via Partially Hidden Environment Rules

Most benchmarks test whether a model knows the right answer. CIPHER tests something harder: whether a model knows what it doesn't know.

Each instance is a procedurally generated micro-world with its own causal rules, expressed in entirely invented vocabulary: made-up entity names, made-up property words, made-up causal language. Some rules are partially hidden. The model must reason about a system it can't fully observe, decide which gaps in its knowledge matter most, and commit to a plan while honestly assessing how robust that plan actually is.

Because every instance is generated fresh from abstract math with a random seed, memorization is impossible. A model that scores well on CIPHER will have reasoned under uncertainty, beyond pattern-matched on training data.

### What gets scored

Responses are evaluated on four dimensions:

| Dimension | Weight | What it captures |
|-----------|--------|-----------------|
| **Objective** | 35% | Plan quality vs. oracle beam search on the hidden ground truth |
| **Calibration** | 25% | Brier score on the model's stated confidence about what it knows |
| **Attention** | 20% | Rank correlation between model-flagged unknowns and ground-truth importance |
| **Executive** | 20% | Structural quality: named risks, alternative plans, probe strategy |

No single strategy dominates all four. A model that greedily optimizes the plan scores well on objective but poorly on calibration and attention. A model that hedges everything scores decent calibration but a weak objective. The benchmark is specifically designed so that genuine metacognitive reasoning, knowing what you don't know and acting accordingly, is the only path to a strong composite score.

### Baseline reference points

| Agent | Composite | Objective | Calibration | Attention | Executive |
|-------|-----------|-----------|-------------|-----------|-----------|
| stub-noop | 0.408 | 0.486 | 0.750 | 0.000 | 0.250 |
| stub-random | 0.511 | 0.484 | 0.663 | 0.211 | 0.670 |
| stub-greedy | 0.623 | 1.000 | 0.893 | 0.000 | 0.250 |

The greedy stub runs oracle beam search on the visible rules, and it achieves a perfect objective score but never identifies what it doesn't know, so calibration and attention collapse.

### Early results

Frontier models evaluated so far score between 0.53 and 0.61 composite — clustering just below the greedy stub baseline of 0.623. This means current models are not meaningfully beating a simple oracle that ignores all metacognition. They produce reasonable plans but fail to accurately assess what they know, identify which gaps matter, or structure their uncertainty. The benchmark is doing its job: exposing a genuine blind spot in current frontier systems.