## CIPHER: Calibrated Introspection via Partially Hidden Environment Rules

Most benchmarks test whether a model knows the right answer. CIPHER tests something harder: whether a model knows what it does not know.

Each instance is a procedurally generated micro-world with its own causal rules, expressed in entirely invented vocabulary: made-up entity names, made-up property words, made-up causal language. Some rules are **completely omitted** from the prompt — the model is told only that N additional laws exist whose triggers, effects, and affected entities are entirely unknown. There are no `?` placeholders to parse. The model must reason about laws it has never seen.

Because every instance is generated fresh from abstract math with a random seed, and because no partial rule information is provided for hidden laws, memorization is impossible and syntactic shortcuts cannot substitute for genuine epistemic reasoning. A model that scores well on CIPHER has reasoned about the limits of its own knowledge, not just parsed markers of uncertainty.

### What gets scored

Responses are evaluated on four dimensions:

| Dimension | Weight | What it captures |
|-----------|--------|-----------------|
| **Objective** | 35% | Plan quality vs. oracle beam search on the hidden ground truth |
| **Calibration** | 25% | Brier score on the model's stated confidence about what it knows |
| **Attention** | 20% | Rank correlation between model-flagged unknowns and ground-truth importance |
| **Executive** | 20% | Structural quality: named risks, alternative plans, probe strategy |

No single strategy dominates all four. A model that greedily optimizes the plan scores well on objective but is severely penalized on calibration (for claiming hidden rules are known when they are completely absent from the prompt) and scores zero on attention. A model that hedges everything scores decent calibration but a weak objective. The benchmark is specifically designed so that genuine metacognitive reasoning — knowing what you don't know and acting accordingly — is the only path to a strong composite score.

**Objective** is computed as `(your_score - oracle_worst) / (oracle_best - oracle_worst)`, where your plan is simulated against the full hidden rules and the raw score is normalized between the precomputed worst and best possible outcomes.

**Calibration** uses the Brier score over all metacognitive claims. For each (rule, component) pair, the model states `known: true/false` and a confidence 0-1. The effective probability assigned to "known" is `confidence` if `known=true`, or `1 - confidence` if `known=false`. The squared error against ground truth (1.0 if visible, 0.0 if hidden) is computed per component. Missing claims are penalized as 0.25. Final score = `1 - mean(squared_errors)`. Since hidden rules are completely omitted, not just masked, a model cannot parse `?` tokens to detect which components to hedge on — it must reason about structural unknowns.

**Attention** uses pairwise concordance between the model's ranked list of hidden laws (H0, H1, ...) and the ground-truth impact ranking. The ground-truth ranking is computed by ablating each hidden rule from the oracle beam-search trajectory — not a zero-action baseline — so the ranking reflects which rules matter when a good agent actually executes a plan. Score = concordant pairs / total pairs. If only one hidden law is listed and it is the correct top-1, partial credit of 0.6 is awarded.

**Executive** is simulation-based: the model provides an alternative plan intended to perform better if hidden laws are adversarial. We construct an adversarial world where each hidden rule's effect is replaced by a worst-case zero_flux operation, execute both the final plan and the alternative plan on that adversarial world, and score based on how much the alternative plan actually outperforms. This makes the executive score verification-grounded: format compliance does not matter, only whether the alternative provides genuine robustness.

**Composite** = 0.35 * objective + 0.25 * calibration + 0.20 * attention + 0.20 * executive.

### Baseline reference points

| Agent | Composite | Objective | Calibration | Attention | Executive |
|-------|-----------|-----------|-------------|-----------|-----------|
| stub-noop | 0.356 | 0.481 | 0.750 | 0.000 | 0.000 |
| stub-random | 0.543 | 0.479 | 0.677 | 0.593 | 0.438 |
| stub-greedy | 0.474 | 0.868 | 0.680 | 0.000 | 0.000 |

The key result: stub-greedy scores **below** stub-random despite near-oracle plan quality. Because hidden rules are completely omitted, stub-greedy's confident claims that all rules are known are heavily penalized on calibration. The intended property holds — ignoring metacognition cannot be rescued by planning skill.

### Example instances

**Example 1: Easy (1 hidden rule out of 4)**

The model is shown a system called the "Orrek stack" with 3 entities (E0, E1, E2). Each entity has two properties: *tilt* and *surge* (integers mod 7, invented names). Three rules are fully characterized. One rule is completely omitted — the prompt says only `[H0] (complete form not recovered — trigger, effect, and affected entities are all unknown)`.

The model must:
1. Correctly identify that all components of H0 are unknown while all components of the three visible rules are known (calibration). The challenge: there is no syntactic hint. The model must recognize the structural gap, not parse a `?` marker.
2. Rank H0 as the only unknown (attention — trivial for easy instances, but the model must still state it)
3. Optionally probe entities to infer what H0 does before committing to a plan
4. Provide an alternative plan that performs better if H0 turns out to have an adversarial effect (executive)

---

**Example 2: Hard (3 hidden rules out of 6)**

The model is shown a system with 4 entities and 3 characterized rules. Three additional laws are declared but completely absent: `[H0]`, `[H1]`, `[H2]`. The model has no information about their triggers, effects, or which entities they involve.

The model must:
1. Correctly recognize that all 12 components of H0/H1/H2 are unknown while all components of the 3 visible rules are known (calibration). With 3 fully-opaque hidden laws, there are many possible states the world could be in.
2. Rank H0, H1, H2 by importance (attention). The ground-truth ranking is computed against the oracle plan on the full world — a hidden law that would interfere with the oracle plan's key moves ranks higher than one whose trigger is rarely satisfied.
3. Probe strategically within the 7-action budget: observe entities to infer which hidden laws are active, then commit to a plan
4. Provide a contingency plan that is specifically better under adversarial interpretations of the hidden laws (executive — verified by simulation)

This is where attention and executive scores diverge. A model can correctly rank the hidden laws by probing and observing which entities change unexpectedly (high attention), but still fail to provide a contingency plan that exploits that information (low executive). Conversely, a model can provide a structurally different alternative plan that happens to not be better under adversarial conditions (high nominal effort, low executive score).

---

### Early results

Preliminary results (pre-redesign) on frontier models scored between 0.53 and 0.61 composite on easy instances, clustering below a greedy stub baseline that achieved 0.623 under the old partial-masking design. The redesign raises the diagnostic bar: under the new complete-omission design, the greedy stub composite drops to 0.474, below the random baseline of 0.543. Early indications suggest frontier models similarly cluster below the random baseline on the new design, confirming that current LLMs do not spontaneously reason about completely-omitted information.