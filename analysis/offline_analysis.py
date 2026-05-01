"""
CIPHER offline analysis v2 — per-instance data from results_jsons/.

Outputs:
  1. Empirical bootstrap CIs (resample actual instances)
  2. Paired sign test for same-instance comparisons
  3. Construct validity: Pearson r with public benchmarks
  4. Weighting robustness
  5. Difficulty breakdown with empirical bootstrap
  6. Parse failure analysis per model
  7. Per-instance dimension correlations

Run: python3 analysis/offline_analysis.py
"""

import json, math, random, sys, os

random.seed(42)

_out = open("analysis/offline_results.txt", "w")
_print = __builtins__.print if hasattr(__builtins__, "print") else __builtins__["print"]

def print(*args, **kwargs):
    _print(*args, **{**kwargs, "file": _out})
    _print(*args, **{**kwargs, "file": sys.stdout})

# ---------------------------------------------------------------------------
# Model → file mapping
# ---------------------------------------------------------------------------

FILE_MAP = {
    "GPT-5.4 Nano":             "gpt54nano.json",
    "GPT-5.4 mini":             "gpt54mini.json",
    "GPT-5.4":                  "gpt54.json",
    "Gemini 3 Flash Preview":   "geminiflash.json",
    "Gemini 3.1 Pro Preview":   "geminipro.json",
    "Claude Sonnet 4.6":        "sonnet46.json",
    "Claude Opus 4.7":          "opus46.json",
    "Claude 4.5 Haiku":         "haiku45.json",
    "DeepSeek V3.2":            "deepseek32.json",
    "Qwen 3 Next 80B Instruct": "qwen.json",
    "Gemma 4 31B":              "gemma4.json",
}

PUBLIC_BENCHMARKS = {
    # model_name: (MMLU, GPQA_Diamond, MATH_500)
    "GPT-5.4 mini":            (0.8455, 0.8308, 0.700),
    "GPT-5.4":                 (0.8748, 0.9167, 0.847),
    "GPT-5.4 Nano":            (0.7717, 0.7753, None),
    "Claude Sonnet 4.6":       (0.8734, 0.8561, 0.820),
    "Claude Opus 4.7":         (0.8987, 0.8990, 0.853),
    "Claude 4.5 Haiku":        (0.7872, 0.7222, None),
    "Gemini 3 Flash Preview":  (None,   None,   0.830),
    "Gemini 3.1 Pro Preview":  (0.9099, 0.9545, 0.910),
    "DeepSeek V3.2":           (0.8492, 0.8030, 0.883),
    "Qwen 3 Next 80B Instruct":(None,   None,   0.869),
    "Gemma 4 31B":             (None,   None,   0.730),
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
# Load per-instance scores from each model's run JSON
# ---------------------------------------------------------------------------

_jsons_dir = os.path.join(os.path.dirname(__file__), "results_jsons")

def load_per_instance(filename):
    """Return list of 1000 score dicts, one per instance, in order."""
    with open(os.path.join(_jsons_dir, filename)) as f:
        d = json.load(f)
    rows = []
    for sub in d["subruns"]:
        dr = sub["results"][0]["dictResult"]
        rows.append({
            "composite":   float(dr.get("composite",   0.0)),
            "objective":   float(dr.get("objective",   0.0)),
            "calibration": float(dr.get("calibration", 0.0)),
            "attention":   float(dr.get("attention",   0.0)),
            "executive":   float(dr.get("executive",   0.0)),
            "parse_error": set(dr.keys()) == {"executive","objective","composite","attention","calibration"},
        })
    return rows

MODEL_DATA = {name: load_per_instance(fname) for name, fname in FILE_MAP.items()}

def mean_dim(rows, dim, difficulty=None):
    subset = rows if difficulty is None else [r for r, d in zip(rows, DIFFICULTY) if d == difficulty]
    return sum(r[dim] for r in subset) / max(len(subset), 1)

# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

N_BOOT = 10000

def bootstrap_paired_ci(rows_a, rows_b, dim, difficulty=None):
    """Empirical bootstrap CI on mean paired diff A-B for a given dim."""
    pairs = [(a[dim] - b[dim]) for a, b, diff in zip(rows_a, rows_b, DIFFICULTY)
             if difficulty is None or diff == difficulty]
    n = len(pairs)
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

def sign_test_p(rows_a, rows_b, dim):
    """Sign test p-value: H0 median paired diff = 0."""
    diffs = [a[dim] - b[dim] for a, b in zip(rows_a, rows_b)]
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n   = pos + neg
    if n == 0:
        return 1.0
    z = (pos - n / 2) / math.sqrt(n / 4)
    return 2 * (1 - _norm_cdf(abs(z))), z

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def pearsonr(xs, ys):
    n  = len(xs)
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

# ---------------------------------------------------------------------------
# 1. Empirical bootstrap CIs — central finding
# ---------------------------------------------------------------------------

section("1. EMPIRICAL BOOTSTRAP CIs — CENTRAL FINDING")
print(f"  (n_boot={N_BOOT}, seed=42, paired by instance)\n")

pairs = [
    ("GPT-5.4 mini",    "GPT-5.4",       "GPT inversion: GPT-5.4 mini vs GPT-5.4"),
    ("Claude Sonnet 4.6","Claude Opus 4.7","Claude inversion: Sonnet 4.6 vs Opus 4.7"),
]

for name_a, name_b, label in pairs:
    ra = MODEL_DATA[name_a]
    rb = MODEL_DATA[name_b]
    print(f"\n{label}")
    print(f"  {'Dim':<14} {'A':>6} {'B':>6} {'Diff':>7}  {'95% CI':>18}  {'z':>5}  {'sign-p':>8}  sig")
    print(f"  {'-'*72}")
    for dim in DIMS:
        ma  = mean_dim(ra, dim)
        mb  = mean_dim(rb, dim)
        diff, lo, hi, z = bootstrap_paired_ci(ra, rb, dim)
        sp, sz = sign_test_p(ra, rb, dim)
        sig = sig_label(z)
        print(f"  {dim:<14} {ma:>6.3f} {mb:>6.3f} {diff:>+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {z:>5.1f}  {sp:>8.4f}  {sig}")

# ---------------------------------------------------------------------------
# 1b. Calibration vs executive dissociation
# ---------------------------------------------------------------------------

section("1b. CALIBRATION vs EXECUTIVE DISSOCIATION (bootstrap)")

print("\nGPT family: smaller models have higher calibration AND higher executive")
for dim in ["calibration", "executive"]:
    diff, lo, hi, z = bootstrap_paired_ci(MODEL_DATA["GPT-5.4 mini"], MODEL_DATA["GPT-5.4"], dim)
    print(f"  GPT mini - GPT full | {dim:<14}: {diff:+.3f}  [{lo:+.3f}, {hi:+.3f}]  z={z:.1f}  {sig_label(z)}")

print("\nClaude family:")
for dim in ["calibration", "executive"]:
    diff, lo, hi, z = bootstrap_paired_ci(MODEL_DATA["Claude Sonnet 4.6"], MODEL_DATA["Claude Opus 4.7"], dim)
    print(f"  Sonnet - Opus       | {dim:<14}: {diff:+.3f}  [{lo:+.3f}, {hi:+.3f}]  z={z:.1f}  {sig_label(z)}")

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
    print(f"  {b:>22}", end="")
print()
print(f"  {'-'*82}")

for dim in DIMS:
    print(f"  {dim:<14}", end="")
    for i, b in enumerate(bench_names):
        valid = [n for n in common if PUBLIC_BENCHMARKS[n][i] is not None]
        if len(valid) < 3:
            print(f"  {'N/A':>22}", end="")
            continue
        xs = [mean_dim(MODEL_DATA[n], dim) for n in valid]
        ys = [PUBLIC_BENCHMARKS[n][i] for n in valid]
        r, p = pearsonr(xs, ys)
        sig = "*" if p < 0.05 else ""
        label = f"{r:+.3f} (p={p:.2f},n={len(valid)}){sig}"
        print(f"  {label:>22}", end="")
    print()

print("\n  * p < 0.05. Low |r| with MMLU/GPQA = CIPHER measures something new.")
print("  Negative r for calibration/executive = larger models score LOWER on CIPHER.")

print("\n  Within-CIPHER dimension correlations (across models, per-instance means):")
for d1, d2 in [("calibration","executive"),("objective","calibration"),("objective","executive")]:
    xs = [mean_dim(MODEL_DATA[n], d1) for n in common]
    ys = [mean_dim(MODEL_DATA[n], d2) for n in common]
    r, p = pearsonr(xs, ys)
    print(f"    {d1} vs {d2}: r={r:+.3f} (p={p:.2f})")

print("\n  Per-instance dimension correlations (pooled across all models × instances):")
all_rows = [r for rows in MODEL_DATA.values() for r in rows]
for d1, d2 in [("calibration","executive"),("objective","calibration"),("objective","executive")]:
    xs = [r[d1] for r in all_rows]
    ys = [r[d2] for r in all_rows]
    r, p = pearsonr(xs, ys)
    print(f"    {d1} vs {d2}: r={r:+.3f} (p={p:.4f}, n={len(all_rows)})")

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

all_model_names = list(MODEL_DATA.keys())

print(f"\n  {'Model':<28}", end="")
for scheme in weight_schemes:
    print(f"  {scheme[:10]:>10}", end="")
print()
print(f"  {'-'*72}")

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
        print(f"  {score:.3f}(#{rank})", end="")
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
    ra = MODEL_DATA[name_a]
    rb = MODEL_DATA[name_b]
    print(f"\n  {family} family: {name_a} - {name_b} composite diff")
    print(f"  {'Stratum':<8} {'A':>6} {'B':>6} {'Diff':>7}  {'95% CI':>18}  {'z':>5}  sig")
    print(f"  {'-'*62}")
    for stratum in strata:
        ma   = mean_dim(ra, "composite", stratum)
        mb   = mean_dim(rb, "composite", stratum)
        diff, lo, hi, z = bootstrap_paired_ci(ra, rb, "composite", stratum)
        sig  = sig_label(z)
        print(f"  {stratum:<8} {ma:>6.3f} {mb:>6.3f} {diff:>+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {z:>5.1f}  {sig}")

    print(f"\n  {family} family: calibration vs executive diff per stratum")
    print(f"  {'Stratum':<8} {'cal diff':>10}  {'exec diff':>10}")
    print(f"  {'-'*32}")
    for stratum in strata:
        cal_diff,  *_ = bootstrap_paired_ci(ra, rb, "calibration", stratum)
        exec_diff, *_ = bootstrap_paired_ci(ra, rb, "executive",   stratum)
        print(f"  {stratum:<8} {cal_diff:>+10.3f}  {exec_diff:>+10.3f}")

# ---------------------------------------------------------------------------
# 5. Parse failure analysis
# ---------------------------------------------------------------------------

section("5. PARSE FAILURE ANALYSIS")

print(f"\n  {'Model':<28} {'Failures':>9} {'Rate':>7} {'Mean composite (valid only)':>28}")
print(f"  {'-'*72}")

for name in all_model_names:
    rows   = MODEL_DATA[name]
    fails  = sum(1 for r in rows if r["parse_error"])
    valid  = [r for r in rows if not r["parse_error"]]
    rate   = fails / len(rows)
    mv     = sum(r["composite"] for r in valid) / max(len(valid), 1)
    print(f"  {name:<28} {fails:>9}  {rate:>6.1%}  {mv:>28.3f}")

# ---------------------------------------------------------------------------
# 6. Score variance across models
# ---------------------------------------------------------------------------

section("6. SCORE VARIANCE (std dev per model, composite)")

print(f"\n  {'Model':<28} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'p10':>6} {'p90':>6}")
print(f"  {'-'*65}")

for name in all_model_names:
    rows = MODEL_DATA[name]
    vals = sorted(r["composite"] for r in rows)
    n    = len(vals)
    mean = sum(vals) / n
    std  = math.sqrt(sum((v - mean)**2 for v in vals) / n)
    p10  = vals[int(0.10 * n)]
    p90  = vals[int(0.90 * n)]
    print(f"  {name:<28} {mean:>6.3f} {std:>6.3f} {vals[0]:>6.3f} {vals[-1]:>6.3f} {p10:>6.3f} {p90:>6.3f}")

print()
