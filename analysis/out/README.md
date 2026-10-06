# CIPHER re-analysis outputs (no new model runs)

Regenerate everything: `.venv/bin/python analysis/run_reanalysis.py` (both scorers, ~35 min cold; cached afterwards).
`v1/` = scorer as submitted (`cipher/scorer.py`). `v2/` = label-normalised scorer (`cipher/scorer_v2.py`).
Objective and executive are identical in v1 and v2; only calibration and attention change.
`cache/` is not included in the zip (rebuilt automatically).

## Why v2 exists (two scorer bugs, both in calibration)

1. **Label matching.** v1 matched claims to ground truth by exact internal rule names (`R0`...). The prompt labels
   hidden laws `H0, H1, ...` and visible rules `[R<idx>] <word> <greek>`. Under v1, hidden-rule claims matched 0% of
   the time for every model; visible claims matched only when written as `R0` rather than `α`/`Canon α`. v1 calibration
   = 0.750 + (share of R-labelled visible claims) x confidence: Opus 4.6 (99% R-labels) 0.902, Haiku 4.5 (0%) exactly 0.750.
   v2 maps every label the prompt shows (`R0`, `α`, `Canon α`, `R0/α`, `[H0]`, `H0_effect_kind`, mis-encoded `Î±`) to the
   internal name; 99-100% of claims map for every model (`v2/p1/inventory.csv`). Duplicate claims for one
   (rule, component) are scored once.
2. **Confidence direction.** The schema never says what `confidence` means. v1 reads it as confidence in the stated
   claim, so `known:false, confidence:0.05` = P(known) 0.95. Every model marks hidden components `known:false` with
   confidence near 0 (median 0.00-0.20), i.e. "how much I know this". With labels fixed but v1's reading, correct answers
   get the maximum penalty. v2: the boolean sets the side, confidence the distance from 0.5
   (P(known) = 0.5 +/- |conf - 0.5|). Sensitivity under the literal and pure-P(known) readings:
   `v2/p1/calibration_confidence_semantics.csv` (symmetric vs P(known) differ by <= 0.003 except GPT-5.5: 1.000 vs 0.939).

**Result:** v2 calibration is 0.92-1.00 for all 16 models. Calibration measures whether a model reports visible = known,
hidden = unknown, which every model does; it is at ceiling and does not discriminate models (C3 confirmed).
Stubs in v2 use prompt labels only (`scripts/evaluate.py`, `STUB_LABELS = "prompt"`); their scores are unchanged.

## Other findings not visible from the CSV names

- **Table 5 = mean over saved responses**, not n=1,000. Missing of 1,000: Gemini 2.5 Pro 451, Gemini 2.5 Flash 245,
  Opus 4.7 162, Opus 4.6 137, Sonnet 4.6 52, GPT-5.4 52, Qwen 54, Gemma 20, Haiku 5, Nano 1, others 0. Failures scored 0 at
  run time, then deleted from the export by `clean_and_rebuild.py` (raw text lost). `v*/p1/`.
- **Gemini 2.5 Pro received mis-encoded prompts** (77 replies echo `Î±`). Every headline statistic is also reported excluding it.
- **Decoding settings** (temperature, max tokens) are not recorded in the Kaggle exports.
- **Stored difficulty labels were wrong for 35-55% of rows** (assigned by file position). All by-difficulty numbers here use
  prompt-matched instance ids. `v*/p2_8/by_difficulty.csv`.
- **Table 4 was not paired by instance**: the old script zipped files positionally (939/948 pairs mismatched).
  `v*/p2_8/paired_comparisons.csv` has both; "instance" rows are correct.
- **Probes return nothing** (single-turn), so A rewards declaring observe actions. Executive spread across models is
  mostly A (0.00-0.95); B ranges 0.14-0.38.
- **Shared budget is not the explanation**: final plans re-run without probes change objective by <= 0.016 for every LLM
  (`v*/p2_1`). Separate-budget dry run moves r by [-0.01, +0.03] (`v1/p3_dryrun`).
- **B's anti-correlation is an artefact of its reference** (`v*/p2_2`): B compares the contingency with the model's OWN
  plan. Scored against oracle bounds on the adversarial world (B_abs), objective~B_abs is r=+0.76 (rho=+0.71).
  Executive with B_abs: r=-0.64 (rho=-0.44, p=0.09), carried by A.
- **zero_flux choice does not matter** (`v*/p2_3`): model B rankings under zero_flux / true / random worlds agree at
  rho 0.97-0.99. Max-damage reorders (rho 0.67-0.72) and B_maxdmg is uncorrelated with objective (r=-0.13). Max-damage is
  exhaustive for 1-2 hidden rules; for 3, pairwise block-coordinate descent (exact on 87.5% of 40 validation cases,
  mean gap 0.18 objective points).
- **Attention does not discriminate models** (`v*/p2_4`): no LLM beats the fixed given-order ranking (H0, H1, ...); its
  advantage over random (0.701 vs 0.500 on medium+hard) is a tie-breaking artefact (35% of medium/hard instances have all
  impacts tied; H-order vs impact among strict pairs 51%, p=0.58).
- **GPQA values** in `p2_8/construct_validity.csv` equal MMLU-Pro for Haiku 4.5, GPT-5.4 Nano, GPT-5.4 mini,
  DeepSeek V3.2; verify against Vals.AI before using.

## Folder map (same in v1/ and v2/)

| folder | content |
|---|---|
| p1 | inventory, Table 5 reproduction, (v2) label audit + calibration semantics |
| p2_1 | executive decomposition (A, B, sub-parts, buckets, #probes), probe-free objective |
| p2_2 | contingency vs fixed reference (B_abs, vs no-op) |
| p2_3 | B under zero_flux / true / random(20) / max-damage; rank agreement; correlations |
| p2_4 | attention baselines (given/reverse/random), per-model, composite without attention |
| p2_8 | model-level tables (valid-only / missing-as-zero / common-instances), all correlations, leave-out, paired comparisons, by-difficulty, construct validity, weighting robustness, floor distances |
| v1/p3_dryrun | separate-probe-budget pipeline test on original replies |
