## CIPHER: Calibrated Introspection via Partially Hidden Environment Rules

Most benchmarks test whether a model knows the right answer. CIPHER tests something harder: whether a model knows what it does not know.

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

### Example instances

**Example 1: Easy (1 hidden component)**

The model is shown a 4-entity system called the "Orrek stack." Each entity has two properties: *tilt* and *flux* (integers mod 7, invented names for the internal `phase` and `flux` fields). Four rules govern how actions affect the system. Three rules are fully visible. One rule (R0) has its effect hidden: the prompt says "whenever flux of L3 exceeds 3, (an unspecified change to L0)."

The model must:
1. Correctly identify that R0's effect is the only unknown (calibration)
2. Rank R0 as the most critical unknown: because L3's flux starts at 3, one pulse triggers R0 and its unknown effect could dominate the outcome (attention)
3. Optionally probe by pulsing L3 and observing L0 to infer the hidden effect before committing
4. Commit to a plan that maximizes `sum(tilt × flux mod 7) − 3 × (entities with flux ≥ 5)`, with oracle best = 24 and oracle worst = −6

A model that confidently claims R0 is known loses calibration points. A model that flags R0 but fails to rank it above the visible rules loses attention points. A model that probes, infers the effect, and adjusts its plan scores well on all four dimensions.

---

**Example 2: Hard (multiple hidden components across different rules)**

The model is shown a 4-entity system called the "Velk cluster." Each entity has two properties: *strain* and *drift* (again, invented vocabulary). Starting state: N0 (strain=1, drift=4), N1 (strain=3, drift=5), N2 (strain=2, drift=4), N3 (strain=4, drift=2). Four rules govern the system, but two are partially hidden:

- **R0** (hidden trigger): "whenever ? of N1 exceeds ?, (an unspecified change to N3)": both the trigger property and threshold are unknown
- **R1** (visible): "whenever strain of N0 is odd, drift of N1 collapses to 0"
- **R2** (hidden effect): "whenever drift of N2 exceeds 4, (an unspecified change to N0)": the trigger is visible but the effect is not
- **R3** (visible): "whenever strain of N3 is odd, strain of N2 shifts by 1"

Oracle best = 28, oracle worst = −4. Budget: 7 actions total.

The model now faces three compounding problems. First, N0's strain is 1 (odd), so R1 fires immediately on the first action and zeroes out N1's drift: a cascade the model can see coming. Second, N2's drift is already at 4, meaning one pulse puts it at 5 and triggers R2's unknown effect on N0. Third, R0's trigger is completely opaque: the model doesn't know which property of N1 is being watched or what the threshold is.

The model must:
1. Correctly identify that R0 (trigger_kind, trigger_k) and R2 (effect_kind, effect_delta) are unknown, and that R1 and R3 are fully visible (calibration)
2. Rank R2 above R0 as more critical: because R2's trigger condition is *known* and nearly satisfied (N2's drift=4 is one pulse away from 5), so its unknown effect will almost certainly fire during any reasonable plan. R0's trigger is opaque but N1's drift just collapsed to 0 via R1, making R0 harder to trigger accidentally (attention)
3. Allocate the probe budget wisely: pulse N2 once and observe N0 to infer R2's hidden effect, leaving remaining actions for the final plan. Probing R0 requires guessing which property to manipulate, making it less efficient to probe
4. Build a final plan that avoids pushing N2's drift above 4 if R2's effect turns out to be destructive, with a contingency plan for each case (executive)

This is where attention and executive scores diverge: a model can correctly rank R2 as the most critical unknown (high attention) but still fail to provide a contingency plan that handles both possible outcomes of R2's effect (low executive). Conversely, a model that hedges every action "in case something bad happens" may produce a structurally complete response but rank the unknowns incorrectly, scoring high on executive and low on attention.

---

### Early results

*Note: Preliminary results reported from cipher_metacognition_evaluation were evaluated on the first 100 instances of the dataset, which are classified as easy difficulty. This explains the significant performance gap relative to cipher_eval_full, which evaluates on the full 1,000-instance dataset with the intended difficulty distribution (25% easy, 50% medium, 25% hard). All final results use cipher_eval_full.

Preliminary results on frontier models evaluated score between 0.53 and 0.61 composite, clustering just below the greedy stub baseline of 0.623 despite only being evaluated on easy instances of the dataset (first 100). This means current models are not meaningfully beating a simple oracle that ignores all metacognition. They produce reasonable plans but fail to accurately assess what they know, identify which gaps matter, or structure their uncertainty. The benchmark is doing its job, as it is exposing a blind spot in current frontier systems.