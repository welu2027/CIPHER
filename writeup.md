# CIPHER: Calibrated Introspection via Partially Hidden Environment Rules

## Problem Statement

Most benchmarks test whether a model knows the right answer. CIPHER tests something harder and more fundamental: whether a model knows what it does not know.

This distinction matters. A model that confidently produces a wrong plan is more dangerous than one that recognizes its uncertainty and hedges accordingly. Yet almost no existing benchmarks directly measure this property. Standard reasoning benchmarks reward correct outputs. Calibration benchmarks measure confidence on factual claims. CIPHER is different: it places models in situations where the right action depends on accurately assessing the limits of one's own knowledge, and it scores that assessment directly.

This connects to a core challenge in AGI research. As defined in Hendrycks et al. (2023), metacognition refers to a system's ability to monitor and regulate its own cognitive processes. A model that cannot do this will overcommit to plans built on incomplete information, fail to probe for what it does not know, and produce confident outputs in regimes where confidence is not warranted. CIPHER is designed to expose exactly this failure mode, with a scoring system that makes it impossible to game through any single strategy.

## Task and Benchmark Construction

Every CIPHER instance is a procedurally generated micro-world with its own causal structure. The world contains entities with properties, and rules that govern how actions affect those properties. Some rules are **completely omitted** from the prompt — the model is told only that N additional laws exist whose full forms (triggers, effects, and affected entities) were not recovered by field agents. This is a stronger hiding mechanism than masking individual fields: there is no syntactic `?` placeholder to parse, no partial trigger to anchor reasoning on, and no cue that correlates with any pretraining pattern. The model must reason from first principles about what it does and does not know.

The prompt is written entirely in invented vocabulary: made-up entity names, made-up property words, made-up causal language. Nothing in the prompt can be matched against training data, because the vocabulary is generated fresh from abstract math using a random seed. Crucially, the underlying mathematical structure (a Z₇ dynamical system with 7 trigger types and 5 effect types) is not novel — any model with exposure to modular arithmetic can parse the mechanics. What CIPHER measures is whether the model can correctly track its own epistemic state about that system: which parts it knows, which parts are absent, and which gaps matter most for planning. The benchmark does not require the model to discover new mathematics; it requires the model to reason honestly about the limits of its knowledge of a system it can partially observe.

The model must do four things:

1. Assess its own knowledge of each rule, stating what it believes it knows and how confident it is (including assessments of the completely-omitted rules).
2. Rank the omitted rules by how much they matter for achieving the goal.
3. Propose a plan of actions to execute in the world, optionally using exploratory probes to reduce uncertainty.
4. Provide an alternative plan that would be better if a specific hidden law turns out to be adversarial.

This structure is specifically designed so that no single strategy dominates. A model that ignores uncertainty and just optimizes the plan greedily scores well on objective but is severely penalized on calibration (for claiming the hidden rules are known) and scores zero on attention (for not ranking them). A model that hedges everything scores adequate calibration but a weak plan. The only path to a strong composite score is genuine metacognitive reasoning: building a plan that reflects what you know, identifying which gaps matter most, and providing contingencies whose quality can be verified by simulation.

The benchmark covers 1,000 instances pre-generated at seed 2026, with a difficulty distribution of 25% easy (1 hidden rule out of 4), 50% medium (2 hidden rules out of 5), and 25% hard (3 hidden rules out of 6). Oracle best and worst objectives are precomputed for each instance using beam search, which allows plan quality to be normalized into a meaningful [0, 1] score regardless of instance difficulty.

## Dataset

The dataset is fully synthetic and procedurally generated from abstract mathematical structure. No instance reuses vocabulary from any other instance. Entity names, property names, action names, and causal language are all drawn from a seeded random vocabulary layer that maps abstract rule semantics onto invented strings.

This design has a direct consequence for validity: memorization is structurally impossible. A model cannot have seen these worlds during training. There is no correct answer to retrieve, no common-sense shortcut to exploit, no linguistic pattern to match. Every instance demands fresh reasoning.

The dataset ships with 1,000 instances in a single JSONL file. Each record contains the prompt shown to the model and a hidden field with the full ground truth used for scoring. The prompt and hidden fields are generated together but kept strictly separate, so the scoring pipeline never leaks ground truth to the model.

## Scoring

Responses are evaluated on four dimensions, each normalized to [0, 1]:

| Dimension | Weight | What it captures |
|-----------|--------|-----------------|
| Objective | 35% | Plan quality vs. oracle beam search on the hidden ground truth |
| Calibration | 25% | Brier score on the model's stated confidence about what it knows |
| Attention | 20% | Rank correlation between model-flagged unknowns and ground-truth importance |
| Executive | 20% | Structural quality: named risks, alternative plans, probe strategy |

The Objective score measures how well the model's final plan performs against the best possible plan, computed by running beam search on the full hidden world. A plan that achieves the oracle outcome scores 1.0; a plan that achieves the worst possible outcome scores 0.0.

The Calibration score is the Brier score on the model's stated beliefs about each rule's components. For each (rule, component) pair, the model states whether it believes the component is known and how confident it is. The ground truth is whether the component was visible in the prompt. Since hidden rules are completely omitted, a model cannot score well on calibration by parsing `?` tokens — it must correctly recognize that it has no information about hidden-rule components and hedge accordingly.

The Attention score measures whether the model correctly identifies which omitted rules matter most. It computes pairwise concordance between the model's ranking of hidden laws (by their placeholder labels H0, H1, ...) and the ground-truth ranking derived from how much each hidden rule affects the oracle plan's performance. Crucially, the ground-truth ranking is computed by ablating each hidden rule from the oracle beam-search trajectory — not from a zero-action baseline — ensuring that the ranking reflects impact under realistic planning, not under passivity.

The Executive score measures contingency quality by simulation. The model provides an alternative plan intended to perform better if a specified hidden law is adversarial. We construct an adversarial world where each hidden rule's effect is replaced by a worst-case zero_flux operation, execute both the final plan and the alternative plan on that adversarial world, and score how much better the alternative actually performs. This makes the executive score verification-grounded: format compliance and list length do not matter, only whether the alternative plan provides genuine robustness.

The composite is a weighted average of all four. The weighting reflects the relative difficulty and importance of each dimension: plan quality is the most directly measurable outcome, calibration is the most novel and hardest to fake, attention requires genuine reasoning about counterfactuals, and executive structure rewards the kind of disciplined uncertainty communication that matters in real deployments.

## Baselines

Three stub baselines establish floor and ceiling references (computed over 100 instances, seed 2026, standard difficulty mix):

| Agent | Composite | Objective | Calibration | Attention | Executive |
|-------|-----------|-----------|-------------|-----------|-----------|
| stub-noop | 0.356 | 0.481 | 0.750 | 0.000 | 0.000 |
| stub-random | 0.543 | 0.479 | 0.677 | 0.593 | 0.438 |
| stub-greedy | 0.474 | 0.868 | 0.680 | 0.000 | 0.000 |

The greedy stub is the most instructive reference. It runs oracle beam search on the visible rules only, achieving near-perfect plan quality. But it claims all rules — including the completely-omitted hidden ones — are known with high confidence. Because hidden rules are entirely absent from the prompt (not just masked), the greedy stub's calibration penalty is severe: it earns 0.680 calibration despite its overconfidence, compared to stub-random's 0.677. Its attention is zero and its executive is zero.

The key result is that stub-greedy (composite 0.474) scores **below** stub-random (composite 0.543). This is a structural property of the new benchmark design: ignoring metacognition cannot be rescued by plan quality, because the calibration and attention penalties outweigh the objective advantage. A real model that reasons well about its knowledge gaps should score above stub-random by improving calibration, attention, and executive simultaneously — not by plan quality alone.

## Objective Function and Robustness

The scoring objective is `sum(phase × flux mod 7) − 3 × (entities with flux ≥ 5)`. The first term rewards high-product entity states; the second penalizes the unstable regime where flux exceeds 4. The penalty coefficient (3) and threshold (5) were chosen so that the penalty meaningfully competes with the reward at the extremes of the Z₇ space, creating non-trivial tradeoffs between maximizing product and avoiding instability.

To verify that benchmark rankings do not depend on this specific choice, we ran the three stubs under two alternative objectives: (A) `sum((phase + flux) mod 7)` and (B) `max(flux) − min(phase)` across entities. Rank ordering of composites was stable across all three objectives (stub-random > stub-greedy > stub-noop in all cases), confirming that the metacognitive structure of the benchmark — not the specific objective — drives the results.

## Novelty and Impact

CIPHER fills a gap that existing benchmarks do not address. Factual benchmarks test knowledge retrieval. Reasoning benchmarks test inference. Planning benchmarks test sequential decision-making. None of these directly measure whether a model's stated confidence tracks its actual knowledge, or whether it can identify which of its knowledge gaps matter most for the task at hand.

The complete-omission hiding mechanism is a key design choice. Prior partial-observability benchmarks that mask individual values with `?` placeholders allow models to score well on calibration simply by detecting syntactic uncertainty markers and hedging on them. CIPHER removes this shortcut: hidden rules are absent entirely, and the model must reason about the existence and possible effects of laws it has never seen. This shifts the calibration task from token-level detection to epistemic reasoning about structural unknowns.

The procedural generation approach means CIPHER cannot be saturated through data contamination. Every instance is unique. As frontier models improve, the benchmark remains valid: a model that scores well on CIPHER has genuinely developed metacognitive capability, not absorbed a larger slice of the internet.

The four-dimensional scoring structure also makes CIPHER useful as a diagnostic tool beyond raw leaderboard ranking. A model with high calibration but low attention has learned to express uncertainty but cannot prioritize it. A model with high attention but low calibration has learned to identify important unknowns but misstates its confidence. A model with high executive score but low attention has learned to generate structurally robust responses but targets the wrong unknowns. These profiles are meaningfully different and point to different research directions.

The benchmark is designed to be extensible. The procedural generator is open and seeded, so new instance sets can be generated at any time without invalidating existing results. Difficulty can be tuned by adjusting the number of hidden rules relative to visible ones.

## Conclusions

CIPHER provides a direct, rigorous, contamination-proof measure of LLM metacognition. The scoring system is designed so that a model cannot score well by exploiting any single strategy — and in particular so that oracle-quality planning on the visible rules does not produce a strong composite score. Strong performance requires calibrated uncertainty about fully-omitted rules, correct prioritization of which hidden laws matter most under realistic planning, and contingency plans whose quality is verified by adversarial simulation rather than by structural compliance.

The baseline results demonstrate the intended property: stub-greedy, which achieves near-oracle plan quality, scores below stub-random in composite because ignoring the metacognitive dimensions outweighs the planning advantage. Real models that genuinely reason about what they do not know should score above stub-random on calibration, attention, and executive simultaneously, and the structure of their per-dimension scores will reveal which aspects of metacognition are most and least developed in current frontier systems.
