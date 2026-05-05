"""
CIPHER offline analysis v3 — per-instance data from analysis/models/.

Each model JSON has 1000 subruns (one per instance); all bootstrap CIs and
sign tests are computed from actual instance-level scores, not aggregates.

Outputs:
  1.  Empirical bootstrap CIs on paired differences (central finding)
  1b. Calibration vs executive dissociation
  2.  Construct validity: Pearson r with public benchmarks (n up to 16)
  3.  Weighting robustness
  4.  Difficulty breakdown with empirical bootstrap
  5.  Parse failure analysis per model
  6.  Score variance (std dev per model)
  7.  Instance hardness: cross-model agreement on hard instances
  8.  Per-instance dimension correlations (pooled and within-model)
  9.  Model consistency: within-model score entropy

Run: python3 analysis/offline_analysis.py
"""

import json, math, random, sys, os

random.seed(42)

_out = open("analysis/offline_results.txt", "w", encoding="utf-8")
_print = __builtins__.print if hasattr(__builtins__, "print") else __builtins__["print"]

def print(*args, **kwargs):
    _print(*args, **{**kwargs, "file": _out})
    _print(*args, **{**kwargs, "file": sys.stdout})

# ---------------------------------------------------------------------------
# Model → file mapping
# ---------------------------------------------------------------------------

FILE_MAP = {
    "GPT-5.4 Nano":             "gpt_54_nano.json",
    "GPT-5.4 mini":             "gpt_54_mini.json",
    "GPT-5.4":                  "gpt_54.json",
    "GPT-5.5":                  "gpt_55.json",
    "Gemini 3 Flash Preview":   "gemini_flash_preview.json",
    "Gemini 3.1 Pro Preview":   "gemini_pro_preview.json",
    "Gemini 2.5 Pro":           "gemini_25_pro.json",
    "Gemini 2.5 Flash":         "gemini_25_flash.json",
    "Claude Sonnet 4.6":        "claude_sonnet_46.json",
    "Claude Opus 4.6":          "claude_opus_46.json",
    "Claude Opus 4.5":          "claude_opus_45.json",
    "Claude Opus 4.7":          "claude_opus_47.json",
    "Claude 4.5 Haiku":         "claude_haiku_45.json",
    "DeepSeek V3.2":            "deepseek_v3_2.json",
    "Qwen 3 Next 80B Instruct": "qwen_3_next_80b_instruct.json",
    "Gemma 4 31B":              "gemma_4_31b.json",
}

PUBLIC_BENCHMARKS = {
    # model_name: (MMLU, GPQA_Diamond, MATH_500)
    "GPT-5.4 Nano":             (0.7717, 0.7753, None),
    "GPT-5.4 mini":             (0.8455, 0.8308, 0.700),
    "GPT-5.4":                  (0.8748, 0.9167, 0.847),
    "GPT-5.5":                  (None,   None,   None),
    "Gemini 3 Flash Preview":   (None,   None,   0.830),
    "Gemini 3.1 Pro Preview":   (0.9099, 0.9545, 0.910),
    "Gemini 2.5 Pro":           (None,   None,   None),
    "Gemini 2.5 Flash":         (None,   None,   None),
    "Claude Sonnet 4.6":        (0.8734, 0.8561, 0.820),
    "Claude Opus 4.6":          (None,   None,   None),
    "Claude Opus 4.5":          (None,   None,   None),
    "Claude Opus 4.7":          (0.8987, 0.8990, 0.853),
    "Claude 4.5 Haiku":         (0.7872, 0.7222, None),
    "DeepSeek V3.2":            (0.8492, 0.8030, 0.883),
    "Qwen 3 Next 80B Instruct": (None,   None,   0.869),
    "Gemma 4 31B":              (None,   None,   0.730),
}

DIMS = ["composite", "objective", "calibration", "attention", "executive"]

# ---------------------------------------------------------------------------
# Load instances.jsonl for difficulty labels
# ---------------------------------------------------------------------------

_data_paths = [
    "data/instances.jsonl",
    os.path.join(os.path.dirname(__file__), "../data/instances.jsonl"),
]
_instances_path = next(p for p in _data_paths if os.path.exists(p))
with open(_instances_path) as f:
    INSTANCES = [json.loads(l) for l in f if l.strip()]

DIFFICULTY = [inst["difficulty"] for inst in INSTANCES]  # list of 1000

# ---------------------------------------------------------------------------
# Load per-instance scores from each model's JSON
# ---------------------------------------------------------------------------

_jsons_dir = os.path.join(os.path.dirname(__file__), "models")

def load_per_instance(filename):
    """Return list of score dicts (one per valid instance), or None if missing/empty.

    After clean_and_rebuild.py runs, failures are already removed and each
    dictResult carries a 'difficulty' field. If 'difficulty' is absent
    (pre-clean files), fall back to positional alignment with DIFFICULTY.
    """
    path = os.path.join(_jsons_dir, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    if not d or "subruns" not in d:
        return None
    rows = []
    for i, sub in enumerate(d["subruns"]):
        dr = sub["results"][0]["dictResult"]
        if "best_objective" not in dr or float(dr.get("parse_errors", 0.0)) > 0:
            continue  # skip any remaining failures
        rows.append({
            "composite":   float(dr.get("composite",   0.0)),
            "objective":   float(dr.get("objective",   0.0)),
            "calibration": float(dr.get("calibration", 0.0)),
            "attention":   float(dr.get("attention",   0.0)),
            "executive":   float(dr.get("executive",   0.0)),
            "difficulty":  dr.get("difficulty"),   # embedded by clean_and_rebuild
        })
    return rows if rows else None

MODEL_DATA = {
    name: rows
    for name, fname in FILE_MAP.items()
    if (rows := load_per_instance(fname)) is not None
}

loaded = list(MODEL_DATA.keys())
missing = [n for n in FILE_MAP if n not in MODEL_DATA]

print(f"Loaded {len(loaded)} models: {', '.join(loaded)}")
if missing:
    print(f"Skipped (no data): {', '.join(missing)}")

all_model_names = loaded

def mean_dim(rows, dim, difficulty=None):
    subset = rows if difficulty is None else [r for r in rows if r.get("difficulty") == difficulty]
    return sum(r[dim] for r in subset) / max(len(subset), 1)

# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

N_BOOT = 10000

def bootstrap_paired_ci(rows_a, rows_b, dim, difficulty=None):
    """Empirical bootstrap CI on mean paired diff A-B (resamples actual instances)."""
    pairs = [a[dim] - b[dim]
             for a, b in zip(rows_a, rows_b)
             if difficulty is None or a.get("difficulty") == difficulty]
    n   = len(pairs)
    obs = sum(pairs) / n
    boot = sorted(
        sum(pairs[random.randint(0, n-1)] for _ in range(n)) / n
        for _ in range(N_BOOT)
    )
    lo = boot[int(0.025 * N_BOOT)]
    hi = boot[int(0.975 * N_BOOT)]
    sd = (hi - lo) / (2 * 1.96)
    z  = obs / sd if sd > 0 else 0.0
    return obs, lo, hi, z

def bootstrap_mean_ci(rows, dim, difficulty=None):
    """Bootstrap CI on a single model's mean."""
    vals = [r[dim] for r in rows
            if difficulty is None or r.get("difficulty") == difficulty]
    n   = len(vals)
    obs = sum(vals) / n
    boot = sorted(
        sum(vals[random.randint(0, n-1)] for _ in range(n)) / n
        for _ in range(N_BOOT)
    )
    lo = boot[int(0.025 * N_BOOT)]
    hi = boot[int(0.975 * N_BOOT)]
    return obs, lo, hi

def sign_test_p(rows_a, rows_b, dim):
    """Sign test p-value: H0 median paired diff = 0."""
    diffs = [a[dim] - b[dim] for a, b in zip(rows_a, rows_b)]
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n   = pos + neg
    if n == 0:
        return 1.0, 0.0
    z = (pos - n / 2) / math.sqrt(n / 4)
    return 2 * (1 - _norm_cdf(abs(z))), z

def paired_stats(rows_a, rows_b, dim, difficulty=None):
    """Return (delta, win_rate, cohen_dz) from real per-instance paired scores."""
    diffs = [a[dim] - b[dim]
             for a, b in zip(rows_a, rows_b)
             if difficulty is None or a.get("difficulty") == difficulty]
    n     = len(diffs)
    delta = sum(diffs) / n
    win_rate = sum(1 for d in diffs if d > 0) / n
    sd    = math.sqrt(sum((d - delta)**2 for d in diffs) / max(n - 1, 1))
    dz    = delta / sd if sd > 0 else 0.0
    return delta, win_rate, dz

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def pearsonr(xs, ys):
    n  = len(xs)
    if n < 3:
        return 0.0, 1.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx)**2 for x in xs) * sum((y - my)**2 for y in ys))
    r   = num / den if den else 0.0
    t   = r * math.sqrt(n - 2) / math.sqrt(max(1 - r**2, 1e-12))
    p   = 2 * (1 - _norm_cdf(abs(t) * math.sqrt(n) / math.sqrt(n + t**2)))
    return r, p

def spearmanr(xs, ys):
    def rank(vs):
        s = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0] * len(vs)
        for rank_val, idx in enumerate(s, 1):
            r[idx] = rank_val
        return r
    return pearsonr(rank(xs), rank(ys))

def section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def sig_label(z):
    return "***" if abs(z) > 3.29 else ("**" if abs(z) > 2.58 else ("*" if abs(z) > 1.96 else "ns"))

def _check_pair(a, b):
    """Return (rows_a, rows_b) if both loaded, else None."""
    if a not in MODEL_DATA or b not in MODEL_DATA:
        return None
    return MODEL_DATA[a], MODEL_DATA[b]

# ---------------------------------------------------------------------------
# 1. Empirical bootstrap CIs — central finding
# ---------------------------------------------------------------------------

section("1. PAIRED COMPARISONS — EFFECT SIZES (CENTRAL FINDING)")
print(f"  delta=mean(A-B), win_rate=P(A>B), d_z=paired Cohen's d")
print(f"  95% CI from empirical bootstrap (n_boot={N_BOOT}, seed=42)\n")

pairs = [
    ("GPT-5.4 mini",     "GPT-5.4",       "GPT inversion: GPT-5.4 mini vs GPT-5.4"),
    ("Claude Sonnet 4.6","Claude Opus 4.7","Claude inversion: Sonnet 4.6 vs Opus 4.7"),
]

for name_a, name_b, label in pairs:
    p = _check_pair(name_a, name_b)
    if p is None:
        print(f"\n{label}  [SKIPPED — data not loaded]")
        continue
    ra, rb = p
    print(f"\n{label}")
    print(f"  {'Dim':<14} {'A':>6} {'B':>6}  {'delta':>7}  {'95% CI':>18}  {'win%':>5}  {'d_z':>6}")
    print(f"  {'-'*70}")
    for dim in DIMS:
        ma  = mean_dim(ra, dim)
        mb  = mean_dim(rb, dim)
        diff, lo, hi, _ = bootstrap_paired_ci(ra, rb, dim)
        delta, wr, dz   = paired_stats(ra, rb, dim)
        print(f"  {dim:<14} {ma:>6.3f} {mb:>6.3f}  {delta:>+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {wr:>4.0%}  {dz:>+6.3f}")
    print(f"  (d_z: |0.2|=small, |0.5|=medium, |0.8|=large)")
    print(f"\n  [appendix] sign-test z and p-value:")
    for dim in DIMS:
        sp, sz = sign_test_p(ra, rb, dim)
        print(f"    {dim:<14} sign-z={sz:>6.1f}  p={sp:.4f}")

# ---------------------------------------------------------------------------
# 1b. Calibration vs executive dissociation
# ---------------------------------------------------------------------------

section("1b. CALIBRATION vs EXECUTIVE DISSOCIATION")

print(f"\n  {'Comparison':<24} {'Dim':<14}  {'delta':>7}  {'95% CI':>18}  {'win%':>5}  {'d_z':>6}")
print(f"  {'-'*74}")
for label, (na, nb) in [("GPT mini - GPT full", ("GPT-5.4 mini","GPT-5.4")),
                         ("Sonnet - Opus 4.7",   ("Claude Sonnet 4.6","Claude Opus 4.7"))]:
    p = _check_pair(na, nb)
    if not p:
        continue
    for dim in ["calibration", "executive"]:
        diff, lo, hi, _ = bootstrap_paired_ci(*p, dim)
        delta, wr, dz   = paired_stats(*p, dim)
        print(f"  {label:<24} {dim:<14}  {delta:>+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {wr:>4.0%}  {dz:>+6.3f}")

# ---------------------------------------------------------------------------
# 1c. Per-model bootstrap CIs on all five dimensions
# ---------------------------------------------------------------------------

section("1c. PER-MODEL BOOTSTRAP CIs (95%, n_boot=10000)")

print(f"\n  {'Model':<28} {'Dim':<14} {'Mean':>6}  {'95% CI':>18}")
print(f"  {'-'*72}")
for name in all_model_names:
    rows = MODEL_DATA[name]
    for dim in DIMS:
        mu, lo, hi = bootstrap_mean_ci(rows, dim)
        print(f"  {name:<28} {dim:<14} {mu:>6.3f}  [{lo:.3f}, {hi:.3f}]")

# ---------------------------------------------------------------------------
# 2. Construct validity — Pearson r with public benchmarks
# ---------------------------------------------------------------------------

section("2. CONSTRUCT VALIDITY — PEARSON r WITH PUBLIC BENCHMARKS")

bench_names = ["MMLU", "GPQA_Diamond", "MATH_500"]
common = [n for n in PUBLIC_BENCHMARKS if n in MODEL_DATA]
print(f"\n  Models in intersection: {len(common)}")
print(f"  {', '.join(common)}\n")

print(f"  {'CIPHER dim':<14}", end="")
for b in bench_names:
    print(f"  {b:>24}", end="")
print()
print(f"  {'-'*86}")

for dim in DIMS:
    print(f"  {dim:<14}", end="")
    for i, b in enumerate(bench_names):
        valid = [n for n in common if PUBLIC_BENCHMARKS[n][i] is not None]
        if len(valid) < 3:
            print(f"  {'N/A (n<3)':>24}", end="")
            continue
        xs = [mean_dim(MODEL_DATA[n], dim) for n in valid]
        ys = [PUBLIC_BENCHMARKS[n][i] for n in valid]
        r, p = pearsonr(xs, ys)
        rho, rp = spearmanr(xs, ys)
        sig = "*" if p < 0.05 else ""
        label = f"r={r:+.3f},ρ={rho:+.3f}(n={len(valid)}){sig}"
        print(f"  {label:>24}", end="")
    print()

print("\n  * p < 0.05. Both Pearson r and Spearman rho reported.")
print("  Low |r| with MMLU/GPQA = CIPHER measures something distinct.")
print("  Negative r for calibration/executive = larger models score LOWER on CIPHER.")

print("\n  Within-CIPHER dimension correlations across models (aggregate means):")
for d1, d2 in [("calibration","executive"),("objective","calibration"),("objective","executive"),
               ("attention","executive"),("attention","calibration")]:
    xs = [mean_dim(MODEL_DATA[n], d1) for n in all_model_names]
    ys = [mean_dim(MODEL_DATA[n], d2) for n in all_model_names]
    r, p = pearsonr(xs, ys)
    rho, _ = spearmanr(xs, ys)
    sig = "*" if p < 0.05 else ""
    print(f"    {d1} vs {d2}: r={r:+.3f}, rho={rho:+.3f} (p={p:.3f}){sig}")

# ---------------------------------------------------------------------------
# 3. Weighting robustness
# ---------------------------------------------------------------------------

section("3. WEIGHTING ROBUSTNESS")

weight_schemes = {
    "Original  (35/25/20/20)": (0.35, 0.25, 0.20, 0.20),
    "Equal     (25/25/25/25)": (0.25, 0.25, 0.25, 0.25),
    "Plan-heavy(50/20/15/15)": (0.50, 0.20, 0.15, 0.15),
    "Meta-heavy(20/35/25/20)": (0.20, 0.35, 0.25, 0.20),
}

def composite_from_weights(name, w):
    rows = MODEL_DATA[name]
    obj  = mean_dim(rows, "objective")
    cal  = mean_dim(rows, "calibration")
    att  = mean_dim(rows, "attention")
    exe  = mean_dim(rows, "executive")
    return w[0]*obj + w[1]*cal + w[2]*att + w[3]*exe

print(f"\n  {'Model':<28}", end="")
for scheme in weight_schemes:
    print(f"  {scheme[:10]:>12}", end="")
print()
print(f"  {'-'*80}")

scheme_scores = {}
for scheme, w in weight_schemes.items():
    scheme_scores[scheme] = sorted(
        [(composite_from_weights(n, w), n) for n in all_model_names], reverse=True
    )

ranks_by_scheme = {}
for scheme, ranked in scheme_scores.items():
    ranks_by_scheme[scheme] = {name: i+1 for i, (_, name) in enumerate(ranked)}

for name in all_model_names:
    print(f"  {name:<28}", end="")
    for scheme, w in weight_schemes.items():
        score = composite_from_weights(name, w)
        rank  = ranks_by_scheme[scheme][name]
        print(f"  {score:.3f}(#{rank:>2})", end="")
    print()

print("\n  Spearman rank correlation vs Original weights:")
orig_ranks = [ranks_by_scheme["Original  (35/25/20/20)"][n] for n in all_model_names]
for scheme in list(weight_schemes.keys())[1:]:
    alt_ranks = [ranks_by_scheme[scheme][n] for n in all_model_names]
    r, p = spearmanr(orig_ranks, alt_ranks)
    print(f"    vs {scheme}: rho={r:.3f} (p={p:.3f})")

# ---------------------------------------------------------------------------
# 4. Difficulty breakdown — empirical bootstrap per stratum
# ---------------------------------------------------------------------------

section("4. DIFFICULTY BREAKDOWN — EMPIRICAL BOOTSTRAP PER STRATUM")

strata = ["easy", "medium", "hard"]

for family, (name_a, name_b) in [
    ("GPT",    ("GPT-5.4 mini", "GPT-5.4")),
    ("Claude", ("Claude Sonnet 4.6", "Claude Opus 4.7")),
]:
    p = _check_pair(name_a, name_b)
    if p is None:
        print(f"\n  {family}: SKIPPED (data not loaded)")
        continue
    ra, rb = p
    print(f"\n  {family} family: {name_a} - {name_b} composite diff by stratum")
    print(f"  {'Stratum':<8} {'A':>6} {'B':>6}  {'delta':>7}  {'95% CI':>18}  {'win%':>5}  {'d_z':>6}")
    print(f"  {'-'*66}")
    for stratum in strata:
        ma   = mean_dim(ra, "composite", stratum)
        mb   = mean_dim(rb, "composite", stratum)
        diff, lo, hi, _ = bootstrap_paired_ci(ra, rb, "composite", stratum)
        delta, wr, dz   = paired_stats(ra, rb, "composite", stratum)
        print(f"  {stratum:<8} {ma:>6.3f} {mb:>6.3f}  {delta:>+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {wr:>4.0%}  {dz:>+6.3f}")

    print(f"\n  {family} family: calibration vs executive by stratum")
    print(f"  {'Stratum':<8} {'cal delta':>7} {'cal win%':>9} {'cal d_z':>8}  {'exe delta':>7} {'exe win%':>9} {'exe d_z':>8}")
    print(f"  {'-'*64}")
    for stratum in strata:
        cd, cwr, cdz = paired_stats(ra, rb, "calibration", stratum)
        ed, ewr, edz = paired_stats(ra, rb, "executive",   stratum)
        print(f"  {stratum:<8} {cd:>+7.3f} {cwr:>8.0%} {cdz:>+8.3f}  {ed:>+7.3f} {ewr:>8.0%} {edz:>+8.3f}")

# ---------------------------------------------------------------------------
# 5. Parse failure analysis
# ---------------------------------------------------------------------------

section("5. PARSE FAILURE ANALYSIS")

print(f"\n  {'Model':<28} {'Failures':>9} {'Rate':>7} {'Composite (valid)':>18} {'Composite (all)':>16}")
print(f"  {'-'*80}")

for name in all_model_names:
    rows   = MODEL_DATA[name]
    fails  = sum(1 for r in rows if r["parse_error"])
    valid  = [r for r in rows if not r["parse_error"]]
    rate   = fails / len(rows)
    mv     = sum(r["composite"] for r in valid) / max(len(valid), 1)
    ma     = mean_dim(rows, "composite")
    print(f"  {name:<28} {fails:>9}  {rate:>6.1%}  {mv:>18.3f}  {ma:>16.3f}")

# ---------------------------------------------------------------------------
# 6. Score variance across models
# ---------------------------------------------------------------------------

section("6. SCORE VARIANCE (per-instance std dev, all dimensions)")

print(f"\n  {'Model':<28} {'Dim':<14} {'Mean':>6} {'Std':>6} {'p10':>6} {'p25':>6} {'p75':>6} {'p90':>6}")
print(f"  {'-'*80}")

for name in all_model_names:
    rows = MODEL_DATA[name]
    for dim in DIMS:
        vals = sorted(r[dim] for r in rows)
        n    = len(vals)
        mean = sum(vals) / n
        std  = math.sqrt(sum((v - mean)**2 for v in vals) / n)
        p10  = vals[int(0.10 * n)]
        p25  = vals[int(0.25 * n)]
        p75  = vals[int(0.75 * n)]
        p90  = vals[int(0.90 * n)]
        print(f"  {name:<28} {dim:<14} {mean:>6.3f} {std:>6.3f} {p10:>6.3f} {p25:>6.3f} {p75:>6.3f} {p90:>6.3f}")

# ---------------------------------------------------------------------------
# 7. Instance hardness: cross-model agreement
# ---------------------------------------------------------------------------

section("7. INSTANCE HARDNESS — CROSS-MODEL AGREEMENT")

# Only valid for models with all 1000 instances (no failures removed).
full_models = [n for n in all_model_names if len(MODEL_DATA[n]) == len(INSTANCES)]
print(f"\n  Models with all {len(INSTANCES)} valid instances: {len(full_models)}")
print(f"  {', '.join(full_models) if full_models else 'none'}")

if len(full_models) >= 2:
    inst_means = [
        sum(MODEL_DATA[name][i]["composite"] for name in full_models) / len(full_models)
        for i in range(len(INSTANCES))
    ]
    inst_diffs = [MODEL_DATA[full_models[0]][i]["difficulty"] for i in range(len(INSTANCES))]
    diff_numeric = [{"easy": 0, "medium": 1, "hard": 2}[d] for d in inst_diffs]

    r_hard, p_hard = pearsonr(inst_means, diff_numeric)
    print(f"\n  Cross-model mean composite vs difficulty (0=easy,1=med,2=hard):")
    print(f"    r={r_hard:+.3f} (p={p_hard:.4f})")

    for stratum in ["easy", "medium", "hard"]:
        vals = [m for m, d in zip(inst_means, inst_diffs) if d == stratum]
        mu  = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mu)**2 for v in vals) / len(vals))
        print(f"    {stratum:<8}: mean={mu:.3f}  std={std:.3f}  n={len(vals)}")

    threshold_low  = sorted(inst_means)[int(0.05 * len(inst_means))]
    threshold_high = sorted(inst_means)[int(0.95 * len(inst_means))]
    hard_inst = [(i, inst_means[i], inst_diffs[i]) for i in range(len(inst_means)) if inst_means[i] <= threshold_low]
    easy_inst = [(i, inst_means[i], inst_diffs[i]) for i in range(len(inst_means)) if inst_means[i] >= threshold_high]

    print(f"\n  Bottom 5% cross-model composite (n={len(hard_inst)}): difficulty breakdown")
    for stratum in ["easy", "medium", "hard"]:
        print(f"    {stratum}: {sum(1 for _,_,d in hard_inst if d == stratum)}")
    print(f"\n  Top 5% cross-model composite (n={len(easy_inst)}): difficulty breakdown")
    for stratum in ["easy", "medium", "hard"]:
        print(f"    {stratum}: {sum(1 for _,_,d in easy_inst if d == stratum)}")

    print(f"\n  Pairwise model rank correlation on per-instance composite (full models only):")
    print(f"  {'':28}", end="")
    for name in full_models:
        print(f"  {name[:8]:>8}", end="")
    print()
    for name_a in full_models:
        print(f"  {name_a:<28}", end="")
        xs = [MODEL_DATA[name_a][i]["composite"] for i in range(len(INSTANCES))]
        for name_b in full_models:
            ys = [MODEL_DATA[name_b][i]["composite"] for i in range(len(INSTANCES))]
            rho, _ = spearmanr(xs, ys)
            print(f"  {rho:>8.3f}", end="")
        print()
else:
    print("  Skipped — need >= 2 models with no failures for cross-model alignment.")

# ---------------------------------------------------------------------------
# 8. Per-instance dimension correlations
# ---------------------------------------------------------------------------

section("8. PER-INSTANCE DIMENSION CORRELATIONS")

print("\n  Pooled across all loaded models × 1000 instances:")
all_rows = [r for rows in MODEL_DATA.values() for r in rows]
n_pooled = len(all_rows)
pairs_dims = [
    ("calibration", "executive"),
    ("objective",   "calibration"),
    ("objective",   "executive"),
    ("attention",   "executive"),
    ("attention",   "calibration"),
    ("objective",   "attention"),
]
for d1, d2 in pairs_dims:
    xs = [r[d1] for r in all_rows]
    ys = [r[d2] for r in all_rows]
    r, p = pearsonr(xs, ys)
    print(f"    {d1:<14} vs {d2:<14}: r={r:+.4f} (p={p:.4e}, n={n_pooled})")

print("\n  Per-model dimension correlations (within each model's 1000 instances):")
print(f"  {'Model':<28}  {'cal-exe':>8}  {'obj-cal':>8}  {'obj-exe':>8}  {'att-exe':>8}  {'att-cal':>8}")
print(f"  {'-'*74}")
for name in all_model_names:
    rows = MODEL_DATA[name]
    def _r(d1, d2):
        xs = [r[d1] for r in rows]
        ys = [r[d2] for r in rows]
        return pearsonr(xs, ys)[0]
    print(f"  {name:<28}  {_r('calibration','executive'):>+8.4f}  {_r('objective','calibration'):>+8.4f}"
          f"  {_r('objective','executive'):>+8.4f}  {_r('attention','executive'):>+8.4f}"
          f"  {_r('attention','calibration'):>+8.4f}")

# ---------------------------------------------------------------------------
# 9. Model consistency
# ---------------------------------------------------------------------------

section("9. MODEL CONSISTENCY (within-model score entropy)")

print(f"\n  Higher std = model is more instance-sensitive (less consistent)")
print(f"\n  {'Model':<28} {'composite std':>14} {'obj std':>8} {'cal std':>8} {'att std':>8} {'exe std':>8}")
print(f"  {'-'*80}")

def _std(rows, dim):
    vals = [r[dim] for r in rows]
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu)**2 for v in vals) / len(vals))

for name in sorted(all_model_names, key=lambda n: _std(MODEL_DATA[n], "composite"), reverse=True):
    rows = MODEL_DATA[name]
    print(f"  {name:<28} {_std(rows,'composite'):>14.4f} {_std(rows,'objective'):>8.4f}"
          f" {_std(rows,'calibration'):>8.4f} {_std(rows,'attention'):>8.4f} {_std(rows,'executive'):>8.4f}")

print()
_out.close()
