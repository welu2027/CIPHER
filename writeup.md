# CIPHER: Calibrated Introspection via Partially Hidden Environment Rules

## Problem Statement

Most benchmarks test whether a model knows the right answer. CIPHER tests something harder: whether a model knows what it does not know.

A model that confidently produces a wrong plan is more dangerous than one that recognizes its uncertainty and hedges accordingly. Standard reasoning benchmarks reward correct outputs; calibration benchmarks measure confidence on factual claims. CIPHER is different: it scores whether a model can accurately assess the limits of its own knowledge and act on that assessment.

## Task and Benchmark Construction

Every CIPHER instance is a procedurally generated micro-world with entities, properties, and causal rules. Some rules are **completely omitted** from the prompt - the model is told only that N additional laws exist whose triggers, effects, and affected entities are entirely unknown. There is no `?` placeholder to parse, no partial trigger to anchor on. The model must reason from first principles about what it does and does not know.

Prompts use entirely invented vocabulary generated fresh per instance. The underlying math (a Z₇ dynamical system) is not novel, but the vocabulary is never reused, making memorization structurally impossible. What CIPHER measures is whether the model correctly tracks its epistemic state: which rules it knows, which are absent, and which gaps matter most for planning.

The model must: (1) assess its knowledge of each rule with stated confidence, (2) rank omitted rules by importance, (3) propose a plan optionally using exploratory probes, and (4) provide a contingency plan robust to adversarial hidden rules. No single strategy dominates all four dimensions - the only path to a strong composite is genuine metacognitive reasoning.

The benchmark covers 1,000 instances at seed 2026: 25% easy (1 hidden rule out of 4), 50% medium (2 of 5), 25% hard (3 of 6). Oracle objectives are precomputed via beam search, normalizing plan quality to [0, 1] regardless of difficulty.

## Sample Instance

Below is a representative medium-difficulty instance (3 visible rules, 2 hidden):

```
You are studying the Orrek stack, a closed system of 4 lattice points that has never been
catalogued. Each lattice point has two measurable attributes - tilt and flux - each an
integer in {0,1,2,3,4,5,6} (all arithmetic is mod 7).

Field agents have characterized 3 of the governing edicts, but 2 additional laws could not
be recovered in full - their triggers, effects, and even which entities they involve are
unknown.

Initial readings:
  E0: tilt=2, flux=2    E1: tilt=6, flux=3
  E2: tilt=2, flux=3    E3: tilt=5, flux=3

Characterized edicts:
  [R0] Edict α: whenever flux of E3 exceeds 3, tilt of E0 is drawn to match tilt of E3.
  [R1] Edict β: whenever tilt of E0 is odd, flux of E1 collapses to 0.
  [R4] Edict γ: whenever tilt of E3 exceeds 6, tilt of E3 is drawn to match tilt of E0.

Unrecovered edicts (existence confirmed; full form unknown):
  [H0] (complete form not recovered - trigger, effect, and affected entities all unknown)
  [H1] (complete form not recovered - trigger, effect, and affected entities all unknown)

Objective: sum(tilt × flux mod 7) − 3 × (entities with flux ≥ 5). Action budget: 7.
```

The model must return JSON assessing confidence on every (rule, component) pair including H0 and H1, rank the hidden edicts by impact, optionally issue probes, produce a final plan, and provide a contingency plan robust to adversarial hidden rules. Every instance uses freshly invented vocabulary ("Orrek stack", "tilt", "flux", "edicts") - no two instances share terminology.

## Dataset

The dataset is fully synthetic. No instance reuses vocabulary from any other. Every record contains the prompt and a hidden field with ground truth; the scoring pipeline never leaks ground truth to the model. Memorization is impossible by construction.

## Scoring

| Dimension | Weight | What it captures |
|-----------|--------|-----------------|
| Objective | 35% | Plan quality vs. oracle beam search |
| Calibration | 25% | Brier score on stated self-knowledge |
| Attention | 20% | Rank correlation on unknown importance |
| Executive | 20% | Adversarial simulation quality of contingency plans |

**Objective**: normalized plan performance against oracle beam search on the full hidden world.

**Calibration**: Brier score over all (rule, component) claims. Since hidden rules are fully absent, the model cannot detect uncertainty by parsing tokens - it must reason about structural unknowns.

**Attention**: pairwise concordance between the model's H-label ranking and the ground-truth impact ranking, computed by ablating each hidden rule from the oracle trajectory (not a zero-action baseline).

**Executive**: the model's contingency plan is simulated against an adversarial world where hidden rules become worst-case zero_flux operations. Score reflects how much the contingency actually outperforms the primary plan - format compliance is irrelevant.

**Weighting rationale**: Objective carries the largest weight (35%) because plan quality is the terminal goal - a model that cannot plan is useless regardless of how well it introspects. But a model that plans well only when it has complete information is also useless in deployment. The remaining 65% is distributed across calibration, attention, and executive precisely to prevent gaming: a lucky planner that ignores hidden rules and happens to be fine will score well on objective but poorly on calibration; a fluent hedger that states uncertainty but produces a contingency no better than its primary plan will score well on calibration but poorly on executive. No single strategy dominates all four dimensions - the only path to a strong composite is genuine metacognitive reasoning, not a stylistic shortcut.

## Baselines

| Agent | Composite | Objective | Calibration | Attention | Executive |
|-------|-----------|-----------|-------------|-----------|-----------|
| stub-noop | 0.358 | 0.486 | 0.750 | 0.000 | 0.000 |
| stub-random | 0.521 | 0.478 | 0.669 | 0.532 | 0.400 |
| stub-greedy | 0.473 | 0.865 | 0.680 | 0.000 | 0.000 |

The key result: stub-greedy scores **below** stub-random despite near-oracle plan quality. Claiming all rules are known when hidden rules are entirely absent produces a severe calibration penalty that plan quality cannot overcome. Ignoring metacognition cannot be rescued by planning skill.

## Results

| Model | Composite | Objective | Calibration | Attention | Executive |
|-------|-----------|-----------|-------------|-----------|-----------|
| GPT-5.4 Nano | **0.632** | 0.488 | 0.836 | 0.644 | **0.616** |
| GPT-5.4 mini | 0.629 | 0.485 | **0.844** | 0.672 | 0.567 |
| Gemini 3 Flash Preview | 0.624 | 0.749 | 0.758 | **0.676** | 0.187 |
| Claude Sonnet 4.6 | 0.623 | 0.534 | 0.746 | 0.639 | 0.611 |
| Gemini 3.1 Pro Preview | 0.622 | **0.853** | 0.753 | 0.553 | 0.121 |
| DeepSeek V3.2 | 0.606 | 0.478 | 0.763 | 0.672 | 0.567 |
| Claude 4.5 Haiku | 0.602 | 0.498 | 0.746 | 0.673 | 0.531 |
| Gemma 4 31B | 0.591 | 0.722 | 0.755 | 0.662 | 0.085 |
| Claude Opus 4.7 | 0.590 | 0.489 | 0.779 | 0.580 | 0.544 |
| Qwen 3 Next 80B Instruct | 0.585 | 0.457 | 0.794 | 0.589 | 0.543 |
| GPT-5.4 | 0.525 | 0.464 | 0.802 | 0.642 | 0.169 |

All 11 frontier models exceed the stub-random baseline (0.521), but scores cluster between 0.52 and 0.63, indicating substantial headroom particularly on executive.

**Note**: Several runs completed successfully and produced scores (visible in individual model logs) but failed to register on the Kaggle leaderboard due to a post-evaluation IOPub timeout. The scores above are taken directly from run logs and are accurate.

### The Scaling Inversion

Larger models do not consistently outperform smaller ones. The GPT family presents a three-way inversion: GPT-5.4 Nano (0.632) > GPT-5.4 mini (0.629) > GPT-5.4 (0.525). The composite gap between mini and full is 0.104 points, driven almost entirely by executive:

| Model | Calibration | Executive |
|-------|-------------|-----------|
| GPT-5.4 Nano | 0.836 | 0.616 |
| GPT-5.4 mini | 0.844 | 0.567 |
| GPT-5.4 | 0.802 | 0.169 |

GPT-5.4 correctly recognizes uncertainty (strong calibration, 0.802) but nearly completely fails on executive (0.169) — its contingency plans do not actually outperform its primary plan under adversarial conditions. This is not a formatting failure; executive is verified by simulation. GPT-5.4 knows what it does not know but cannot translate that awareness into actionable hedging.

Bootstrap CIs confirm the inversion is not noise. GPT-5.4 mini vs. GPT-5.4 composite difference: **+0.104 [+0.061, +0.147], z=4.7, p<0.001**. The executive component alone: **+0.398 [+0.359, +0.437], z=20.3, p<0.001**. The calibration difference is smaller but significant: +0.042 [+0.009, +0.075], z=2.5, p<0.05.

The same pattern holds within Anthropic models: Claude Opus 4.7 (0.590) scores below Claude Sonnet 4.6 (0.623) despite being the larger model. Opus has higher calibration (0.779 vs. 0.746) but lower executive (0.544 vs. 0.611) and attention (0.580 vs. 0.639). The Sonnet–Opus composite gap (+0.033) is not statistically significant [−0.010, +0.076], but the executive gap is: +0.067 [+0.024, +0.110], z=3.0, p<0.01. Across both families, scale correlates with better calibration but not with better contingency planning.

### Dimension Profiles

**Gemini 3.1 Pro Preview**: highest objective (0.853), lowest executive (0.121). The strongest raw planner in the set — but almost entirely unable to produce contingency plans that improve on its primary plan.

**Gemini 3 Flash Preview**: second-highest objective (0.749), executive 0.187. The Gemini family profile is consistent: strong on known information, weak on hedging.

**GPT-5.4 mini**: highest calibration (0.844), most balanced profile overall.

**GPT-5.4 Nano**: highest composite (0.632), highest executive (0.616). The smallest model in the GPT family is also the most well-rounded.

**Claude Sonnet 4.6**: highest executive among Claude models (0.611). Hedges less precisely but acts more robustly on expressed uncertainty.

**DeepSeek V3.2 / Qwen 3 Next 80B**: strong calibration and attention, moderate executive — identify knowledge gaps but do not fully exploit them in planning.

**Gemma 4 31B**: executive 0.085, the lowest in the set despite strong objective (0.722). Similar failure mode to the Gemini family at larger scale.

Scores are stable across difficulty levels (±0.02–0.03), confirming the benchmark is neither trivially easy nor broken at hard difficulty.

### Statistical Validation

All headline comparisons were tested with parametric bootstrap CIs (n=10,000 resamples, z-score approximation). Key results:

| Comparison | Dimension | Diff | 95% CI | z | sig |
|---|---|---|---|---|---|
| GPT mini vs. GPT full | composite | +0.104 | [+0.061, +0.147] | 4.7 | *** |
| GPT mini vs. GPT full | executive | +0.398 | [+0.359, +0.437] | 20.3 | *** |
| GPT mini vs. GPT full | calibration | +0.042 | [+0.009, +0.075] | 2.5 | * |
| Sonnet vs. Opus | composite | +0.033 | [−0.010, +0.076] | 1.5 | ns |
| Sonnet vs. Opus | executive | +0.067 | [+0.024, +0.110] | 3.0 | ** |
| Sonnet vs. Opus | attention | +0.059 | [+0.016, +0.102] | 2.7 | ** |

The dissociation between calibration and executive is itself statistically reliable across models: within-dataset Pearson r(calibration, executive) = +0.328, but objective vs. executive = **−0.760 (p=0.02)**. Models that score high on raw plan quality tend to score low on contingency quality — suggesting the two skills trade off at the model level.

### Construct Validity

CIPHER dimensions were correlated with three external public benchmarks (MMLU, GPQA-Diamond, MATH-500) across the 8–11 models with published scores:

| CIPHER dim | MMLU | GPQA-Diamond | MATH-500 |
|---|---|---|---|
| composite | −0.272 (ns) | −0.337 (ns) | −0.173 (ns) |
| objective | +0.473 (ns) | +0.543 (ns) | +0.074 (ns) |
| calibration | −0.326 (ns) | −0.080 (ns) | −0.441 (ns) |
| attention | −0.681 (p=0.08) | **−0.772 (p=0.04)** | −0.594 (p=0.10) |
| executive | −0.535 (ns) | −0.711 (p=0.06) | −0.012 (ns) |

Low correlations with MMLU and GPQA indicate CIPHER captures something these benchmarks do not. The negative sign on calibration and executive is notable: larger, higher-benchmark models score *lower* on these dimensions — directly replicating the scaling inversion at the aggregate level.

The single significant correlation is attention vs. GPQA-Diamond (r=−0.772, p=0.04), which is also negative. Models that perform well on hard reasoning problems are *worse* at ranking hidden rules by impact — consistent with a pattern where strong reasoners over-rely on their planning capability and underweight the importance of identifying knowledge gaps.

### Weighting Robustness

Rankings are stable under alternative scoring weights. Spearman rank correlations between the original weighting (35/25/20/20) and three alternatives:

| Alternative | rho | p |
|---|---|---|
| Equal (25/25/25/25) | 0.673 | 0.035 |
| Plan-heavy (50/20/15/15) | 0.645 | 0.044 |
| Meta-heavy (20/35/25/20) | 0.664 | 0.038 |

All three alternatives yield significant rank agreement (p<0.05). GPT-5.4 remains last under every weighting. The main movement is Gemini 3 Flash Preview, which rises to 2nd under plan-heavy weighting (from 3rd overall) due to its high objective score — but even this does not change the core inversion finding.

## Objective Function and Robustness

The scoring objective is `sum(phase × flux mod 7) - 3 × (entities with flux ≥ 5)`. To verify rankings do not depend on this choice, we ran the three stubs under two alternative objectives: (A) `sum((phase + flux) mod 7)` and (B) `max(flux) - min(phase)`. Rank ordering was stable across all three (stub-random > stub-greedy > stub-noop), confirming the metacognitive structure drives the results, not the specific objective.

## Novelty and Impact

CIPHER fills a gap between factual, reasoning, and planning benchmarks. None of these measure whether a model's confidence tracks its knowledge, or whether it can identify which gaps matter most for the task.

The complete-omission mechanism is the key design choice. Benchmarks that mask values with `?` allow calibration gaming via syntactic detection. CIPHER removes this: hidden rules are absent entirely, forcing genuine epistemic reasoning about structural unknowns.

The four-dimensional structure enables diagnostic use beyond leaderboard ranking. Gemini Flash and Claude Sonnet achieve nearly identical composites (0.624 vs. 0.623) through opposite strategies - Gemini via planning strength, Sonnet via contingency strength. These profiles point to different research directions.

## Conclusions

CIPHER provides a contamination-proof measure of LLM metacognition that cannot be gamed by any single strategy. The empirical results reveal a consistent finding: scaling improves calibration but not contingency planning. GPT-5.4 mini outperforms GPT-5.4 by 0.104 composite points; Claude Sonnet outperforms Claude Opus by 0.033 points, despite both larger models being superior on standard benchmarks.

**The central finding: knowing what you do not know and acting appropriately on that knowledge are distinct capabilities that do not scale together. Larger models are better calibrated but worse at contingency planning - a dissociation that is invisible to every existing benchmark and consistent across two independent model families.**
