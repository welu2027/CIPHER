# CIPHER

**Calibrated Introspection via Partially Hidden Environment Rules**

Developed by the Feng Lab at Stevens Institute of Technology, CIPHER evaluates whether LLMs know what they don't know and act on it. Models are placed in procedurally generated causal worlds with invented vocabulary (preventing memorization), where some governing rules are completely omitted from the prompt. A model must plan toward a goal, assess its own confidence in each visible rule, rank which hidden rules pose the greatest risk, and submit a contingency plan that holds under adversarial conditions. 1,000 instances across three difficulty tiers (easy: 1 hidden rule, medium: 2, hard: 3).

**Key finding:** Objective and executive scores are anti-correlated at r = -0.81 across frontier models, showing that planning ability and contingency quality are distinct capabilities that do not scale together.

Please see our [research paper](#) for further details.

---

## Scoring

Models are ranked on a composite score (0-1) across four dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Objective | 35% | Plan quality vs. an oracle solver |
| Calibration | 25% | Accuracy of self-reported confidence (Brier score) |
| Attention | 20% | Rank correlation on hidden rule importance |
| Executive | 20% | Whether its contingency plan outperforms its primary plan under adversarial conditions |

Composite = `0.35 * objective + 0.25 * calibration + 0.20 * attention + 0.20 * executive`
