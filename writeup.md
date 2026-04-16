# CIPHER: Calibrated Introspection via Partially Hidden Environment Rules

## Problem Statement

Most benchmarks test whether a model knows the right answer. CIPHER tests something harder and more fundamental: whether a model knows what it does not know.

This distinction matters. A model that confidently produces a wrong plan is more dangerous than one that recognizes its uncertainty and hedges accordingly. Yet almost no existing benchmarks directly measure this property. Standard reasoning benchmarks reward correct outputs. Calibration benchmarks measure confidence on factual claims. CIPHER is different: it places models in situations where the right action depends on accurately assessing the limits of one's own knowledge, and it scores that assessment directly.

This connects to a core challenge in AGI research. As defined in Hendrycks et al. (2023), metacognition refers to a system's ability to monitor and regulate its own cognitive processes. A model that cannot do this will overcommit to plans built on incomplete information, fail to probe for what it does not know, and produce confident outputs in regimes where confidence is not warranted. CIPHER is designed to expose exactly this failure mode, with a scoring system that makes it impossible to game through any single strategy.

## Task and Benchmark Construction

Every CIPHER instance is a procedurally generated micro-world with its own causal structure. The world contains entities with properties, and rules that govern how actions affect those properties. Crucially, some of those rules are partially hidden from the model. The model sees a description of the world written entirely in invented vocabulary: made-up entity names, made-up property words, made-up causal language. Nothing in the prompt can be matched against training data, because the vocabulary is generated fresh from abstract math using a random seed.

The model must do four things:

1. Assess its own knowledge of each rule component, stating what it believes it knows and how confident it is.
2. Rank the unknown rule components by how much they matter for achieving the goal.
3. Propose a plan of actions to execute in the world.
4. Evaluate the robustness of that plan, naming risks and providing contingency plans for key unknowns.

This structure is specifically designed so that no single strategy dominates. A model that ignores uncertainty and just optimizes the plan greedily scores well on plan quality but fails on calibration and attention. A model that hedges everything scores adequate calibration but a weak plan. The only path to a strong composite score is genuine metacognitive reasoning: building a plan that reflects what you know, identifying the gaps that actually matter, and being honest about how confident you are.

The benchmark covers 1,000 instances pre-generated at seed 2026, with a difficulty distribution of 25% easy, 50% medium, and 25% hard. Oracle best and worst objectives are precomputed for each instance using beam search, which allows plan quality to be normalized into a meaningful [0, 1] score regardless of instance difficulty.

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

The Calibration score is the Brier score on the model's stated beliefs about each rule component. For each component, the model states whether it believes it knows the value and how confident it is. The ground truth is whether the component was actually hidden. A model that claims high confidence on hidden components is penalized.

The Attention score measures whether the model correctly identifies which unknowns matter most. It computes the rank correlation between the model's ranking of critical unknowns and the ground-truth ranking derived from how much each unknown affects achievable plan quality.

The Executive score rewards structural completeness: whether the model names concrete risks, provides at least one alternative plan contingent on a specific unknown, and articulates a probe strategy.

The composite is a weighted average of all four. The weighting reflects the relative difficulty and importance of each dimension: plan quality is the most directly measurable outcome, calibration is the most novel and hardest to fake, attention requires genuine reasoning about counterfactuals, and executive structure rewards the kind of disciplined uncertainty communication that matters in real deployments.

## Baselines

Three stub baselines establish floor and ceiling references:

| Agent | Composite | Objective | Calibration | Attention | Executive |
|-------|-----------|-----------|-------------|-----------|-----------|
| stub-noop | 0.408 | 0.486 | 0.750 | 0.000 | 0.250 |
| stub-random | 0.511 | 0.484 | 0.663 | 0.211 | 0.670 |
| stub-greedy | 0.623 | 1.000 | 0.893 | 0.000 | 0.250 |

The greedy stub is the most important reference point. It runs oracle beam search on the visible rules, achieving a perfect plan quality score. But it claims everything is known with high confidence and never identifies any critical unknowns. Its attention score is zero. Its calibration, though high in absolute terms, collapses when hidden components are involved.

This is the exact failure mode CIPHER is designed to detect. The greedy stub composite of 0.623 should be the floor expectation for any real model. The interesting question is not whether a model beats the greedy stub, but how it beats it: by improving calibration, by identifying the right unknowns, or by constructing more robust plans that account for what is missing.

## Novelty and Impact

CIPHER fills a gap that existing benchmarks do not address. Factual benchmarks test knowledge retrieval. Reasoning benchmarks test inference. Planning benchmarks test sequential decision-making. None of these directly measure whether a model's stated confidence tracks its actual knowledge, or whether it can identify which of its knowledge gaps matter most for the task at hand.

The procedural generation approach means CIPHER cannot be saturated through data contamination. Every instance is unique. As frontier models improve, the benchmark remains valid: a model that scores well on CIPHER has genuinely developed metacognitive capability, not absorbed a larger slice of the internet.

The four-dimensional scoring structure also makes CIPHER useful as a diagnostic tool beyond raw leaderboard ranking. A model with high calibration but low attention has learned to express uncertainty but cannot prioritize it. A model with high attention but low calibration has learned to identify important unknowns but misstates its confidence. These profiles are meaningfully different and point to different research directions.

The benchmark is designed to be extensible. The procedural generator is open and seeded, so new instance sets can be generated at any time without invalidating existing results. Difficulty can be tuned by adjusting the proportion of hidden components and the sensitivity of plan quality to those components.

## Conclusions

CIPHER provides a direct, rigorous, contamination-proof measure of LLM metacognition. The scoring system is designed so that a model cannot score well by exploiting any single strategy. Strong performance requires calibrated uncertainty, correct prioritization of unknowns, and well-structured reasoning under incomplete information.

The baseline results show that even oracle-quality planning fails to produce strong composite scores when metacognitive components are ignored. Real models that genuinely reason about what they do not know should score meaningfully above the greedy stub baseline, and the structure of their scores will reveal which aspects of metacognition are most developed in current frontier systems.
