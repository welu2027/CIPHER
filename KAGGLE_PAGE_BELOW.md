## Motivation

Standard calibration benchmarks only test whether a model says "I don't know." CIPHER tests whether a model can reason about the shape of its own ignorance and act on it. Each instance uses freshly invented vocabulary so knowledge cutoff and memorization cannot inflate scores.

---

## How Scoring Works

Each model response is parsed and scored across four dimensions, then combined into a composite:

- **Objective** - the model's final plan is simulated against an oracle beam search. Score = normalized plan value relative to best and worst possible outcomes.
- **Calibration** - Brier score on the model's stated confidence for each rule component (trigger type, threshold, effect type, effect magnitude). Lower Brier = higher calibration score.
- **Attention** - Spearman rank correlation between the model's ranked list of dangerous hidden rules and the ground-truth impact ranking.
- **Executive** - whether the model's named contingency plan outperforms its primary plan when the hidden rules are set adversarially.

---

## Dataset

1,000 procedurally generated instances across three difficulty tiers. Each instance is seeded and fully reproducible. Invented vocabulary (entity names, attribute names, rule names) is re-randomized per instance so no two instances share surface form.

| Tier | Visible rules | Hidden rules |
|---|---|---|
| Easy | 3 | 1 |
| Medium | 3 | 2 |
| Hard | 3 | 3 |

---

## Closing Remarks

Thank you for running CIPHER. For independently validated scores across all frontier models, check the official leaderboard. Questions, comments, or issues? Share your thoughts in the discussion forum.
