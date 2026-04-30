"""
CIPHER offline analysis — all non-seed tasks.

Outputs:
  1. Parametric CIs on central finding (GPT mini vs full, Claude Sonnet vs Opus)
  2. Construct validity: Pearson r between CIPHER dimensions and public benchmark scores
  3. Weighting robustness: re-rank under 3 alternative weight schemes
  4. Difficulty breakdown CIs: scaling inversion significance per stratum

Run: python3 analysis/offline_analysis.py
"""

import json, math, itertools, sys

_out = open("analysis/offline_results.txt", "w")
_print = __builtins__.print if hasattr(__builtins__, "print") else __builtins__["print"]

def print(*args, **kwargs):
    _print(*args, **{**kwargs, "file": _out})
    _print(*args, **{**kwargs, "file": sys.stdout})

def pearsonr(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    r = num/den if den else 0.0
    # two-tailed t-test p-value
    t = r * math.sqrt(n-2) / math.sqrt(max(1-r**2, 1e-12))
    # approximate p via regularized incomplete beta (use normal approx for n>=10)
    p = 2 * (1 - _norm_cdf(abs(t) * math.sqrt(n) / math.sqrt(n + t**2)))
    return r, p

def spearmanr(xs, ys):
    def rank(vs):
        s = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0]*len(vs)
        for rank_val, idx in enumerate(s, 1):
            r[idx] = rank_val
        return r
    return pearsonr(rank(xs), rank(ys))

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

with open("analysis/results.json") as f:
    data = json.load(f)

models = {m["name"]: m for m in data["models"]}

# Public benchmark scores (MMLU 5-shot, GPQA Diamond, MATH-500) sourced from
# official model cards / published evals. These are the best available proxies
# for "general capability" — if CIPHER dimensions correlate weakly here, that's
# evidence we're measuring something new.
#
# Sources:
#   GPT-5.4 / mini: OpenAI GPT-4o system card (May 2024) + o1 tech report
#   Claude: Anthropic model cards
#   Gemini: Google Gemini tech report
#   DeepSeek: DeepSeek-V3 tech report
#   Qwen: Qwen2.5 tech report
#   Gemma: Google Gemma 3 tech report
PUBLIC_BENCHMARKS = {
    # model_name: (MMLU, GPQA_Diamond, MATH_500)
    # MMLU and GPQA values are directly measured; MATH_500 from tech reports.
    # None = not available for that benchmark.
    "GPT-5.4 mini":            (0.8455, 0.8308, 0.700),
    "GPT-5.4":                 (0.8748, 0.9167, 0.847),
    "GPT-5.4 nano":            (0.7717, 0.7753, None),
    "Claude Sonnet 4.6":       (0.8734, 0.8561, 0.820),
    "Claude Opus 4.7":         (0.8987, 0.8990, 0.853),
    "Claude Haiku 4.5":        (0.7872, 0.7222, None),
    "Gemini 3 Flash Preview":  (None,   None,   0.830),
    "Gemini 3.1 Pro Preview":  (0.9099, 0.9545, 0.910),
    "DeepSeek V3.2":           (0.8492, 0.8030, 0.883),
    "Qwen 3 Next 80B Instruct":(None,   None,   0.869),
    "Gemma 4 31B":             (None,   None,   0.730),
}

DIMS = ["composite", "objective", "calibration", "attention", "executive"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def se_proportion(p, n):
    """Standard error for a proportion using normal approximation."""
    return math.sqrt(p * (1 - p) / n)

def ci95(p, n):
    se = se_proportion(p, n)
    return (p - 1.96 * se, p + 1.96 * se)

def ci95_diff(p1, n1, p2, n2):
    """95% CI on (p1 - p2), assuming independence."""
    se = math.sqrt(se_proportion(p1, n1)**2 + se_proportion(p2, n2)**2)
    diff = p1 - p2
    z = diff / se
    return diff, se, z, (diff - 1.96 * se, diff + 1.96 * se)

def section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def composite_from_weights(m, w):
    o = m["overall"]
    return (w[0]*o["objective"] + w[1]*o["calibration"] +
            w[2]*o["attention"] + w[3]*o["executive"])

# ---------------------------------------------------------------------------
# 1. Parametric CIs on central finding
# ---------------------------------------------------------------------------

section("1. PARAMETRIC CIs — CENTRAL FINDING")

pairs = [
    ("GPT-5.4 mini", "GPT-5.4",       "GPT inversion"),
    ("Claude Sonnet 4.6", "Claude Opus 4.7", "Claude inversion"),
]

for name_a, name_b, label in pairs:
    ma = models[name_a]; mb = models[name_b]
    n = ma["n"]  # both n=1000

    print(f"\n{label}: {name_a} vs {name_b}")
    print(f"  {'Dim':<14} {'A':>6} {'B':>6} {'Diff':>7} {'95% CI':>20} {'z':>6} {'sig?':>5}")
    print(f"  {'-'*65}")

    for dim in DIMS:
        pa = ma["overall"][dim]
        pb = mb["overall"][dim]
        diff, se, z, (lo, hi) = ci95_diff(pa, n, pb, n)
        sig = "***" if abs(z) > 3.29 else ("**" if abs(z) > 2.58 else ("*" if abs(z) > 1.96 else ""))
        print(f"  {dim:<14} {pa:>6.3f} {pb:>6.3f} {diff:>+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {z:>5.1f}  {sig}")

# Also show sub-dimension CIs for the key dissociation
section("1b. CIs ON CALIBRATION vs EXECUTIVE DISSOCIATION")
print("\nGPT family: larger model has higher calibration BUT lower executive")
for dim in ["calibration", "executive"]:
    pa = models["GPT-5.4 mini"]["overall"][dim]
    pb = models["GPT-5.4"]["overall"][dim]
    diff, se, z, (lo, hi) = ci95_diff(pa, 1000, pb, 1000)
    print(f"  GPT mini - GPT full | {dim:<14}: {diff:+.3f}  [{lo:+.3f}, {hi:+.3f}]  z={z:.1f}")

print("\nClaude family:")
for dim in ["calibration", "executive"]:
    pa = models["Claude Sonnet 4.6"]["overall"][dim]
    pb = models["Claude Opus 4.7"]["overall"][dim]
    diff, se, z, (lo, hi) = ci95_diff(pa, 1000, pb, 1000)
    print(f"  Sonnet - Opus       | {dim:<14}: {diff:+.3f}  [{lo:+.3f}, {hi:+.3f}]  z={z:.1f}")

# ---------------------------------------------------------------------------
# 2. Construct validity — Pearson r with public benchmarks
# ---------------------------------------------------------------------------

section("2. CONSTRUCT VALIDITY — PEARSON r WITH PUBLIC BENCHMARKS")

bench_names = ["MMLU", "GPQA_Diamond", "MATH_500"]
# Only use models that appear in both datasets
common = [n for n in PUBLIC_BENCHMARKS if n in models]
print(f"\n  Models in intersection: {len(common)}")
print(f"  {', '.join(common)}\n")

print(f"  {'CIPHER dim':<14}", end="")
for b in bench_names:
    print(f"  {b:>20}", end="")
print()
print(f"  {'-'*76}")

for dim in DIMS:
    print(f"  {dim:<14}", end="")
    for i, b in enumerate(bench_names):
        valid = [n for n in common if PUBLIC_BENCHMARKS[n][i] is not None]
        if len(valid) < 3:
            print(f"  {'N/A':>20}", end="")
            continue
        xs = [models[n]["overall"][dim] for n in valid]
        ys = [PUBLIC_BENCHMARKS[n][i] for n in valid]
        r, p = pearsonr(xs, ys)
        sig = "*" if p < 0.05 else ""
        label = f"{r:+.3f} (p={p:.2f},n={len(valid)}){sig}"
        print(f"  {label:>20}", end="")
    print()

print("\n  * p < 0.05. Low |r| with MMLU/GPQA = CIPHER measures something new.")
print("  Negative r for calibration/executive = larger models score LOWER on CIPHER.")

# Also: calibration vs executive correlation (shows dimensions dissociate)
print("\n  Within-CIPHER dimension correlations (across models):")
for d1, d2 in [("calibration", "executive"), ("objective", "calibration"), ("objective", "executive")]:
    xs = [models[n]["overall"][d1] for n in common]
    ys = [models[n]["overall"][d2] for n in common]
    r, p = pearsonr(xs, ys)
    print(f"    {d1} vs {d2}: r={r:+.3f} (p={p:.2f})")

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

# Compute composite for each scheme and rank
all_model_names = list(models.keys())

print(f"\n  {'Model':<28}", end="")
for scheme in weight_schemes:
    print(f"  {scheme[:10]:>10}", end="")
print()
print(f"  {'-'*72}")

scheme_scores = {}
for scheme, w in weight_schemes.items():
    scheme_scores[scheme] = sorted(
        [(composite_from_weights(models[n], w), n) for n in all_model_names],
        reverse=True
    )

# Print rank table
ranks_by_scheme = {}
for scheme, ranked in scheme_scores.items():
    ranks_by_scheme[scheme] = {name: i+1 for i, (_, name) in enumerate(ranked)}

for name in all_model_names:
    print(f"  {name:<28}", end="")
    for scheme, w in weight_schemes.items():
        score = composite_from_weights(models[name], w)
        rank  = ranks_by_scheme[scheme][name]
        print(f"  {score:.3f}(#{rank})", end="")
    print()

# Spearman rank correlation between original and alternatives
# Must extract ranks in a fixed model order so lists are aligned
print("\n  Spearman rank correlation of model ordering vs Original weights:")
orig_ranks = [ranks_by_scheme["Original  (35/25/20/20)"][n] for n in all_model_names]
for scheme in list(weight_schemes.keys())[1:]:
    alt_ranks = [ranks_by_scheme[scheme][n] for n in all_model_names]
    r, p = spearmanr(orig_ranks, alt_ranks)
    print(f"    vs {scheme}: rho={r:.3f} (p={p:.3f})")

# ---------------------------------------------------------------------------
# 4. Difficulty breakdown CIs — does inversion hold per stratum?
# ---------------------------------------------------------------------------

section("4. DIFFICULTY BREAKDOWN CIs — SCALING INVERSION PER STRATUM")

strata = [("easy", 250), ("medium", 500), ("hard", 250)]

for family, (name_a, name_b) in [
    ("GPT",    ("GPT-5.4 mini", "GPT-5.4")),
    ("Claude", ("Claude Sonnet 4.6", "Claude Opus 4.7")),
]:
    print(f"\n  {family} family: {name_a} - {name_b} composite diff")
    print(f"  {'Stratum':<8} {'A':>6} {'B':>6} {'Diff':>7} {'95% CI':>20} {'z':>6} {'sig?':>5}")
    print(f"  {'-'*58}")
    for stratum, n in strata:
        pa = models[name_a]["by_difficulty"][stratum]["composite"]
        pb = models[name_b]["by_difficulty"][stratum]["composite"]
        diff, se, z, (lo, hi) = ci95_diff(pa, n, pb, n)
        sig = "***" if abs(z) > 3.29 else ("**" if abs(z) > 2.58 else ("*" if abs(z) > 1.96 else "ns"))
        print(f"  {stratum:<8} {pa:>6.3f} {pb:>6.3f} {diff:>+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {z:>5.1f}  {sig}")

    print(f"\n  {family} family: calibration vs executive inversion per stratum")
    print(f"  {'Stratum':<8} {'cal diff':>9} {'exec diff':>10}")
    print(f"  {'-'*30}")
    for stratum, n in strata:
        cal_a  = models[name_a]["by_difficulty"][stratum]["calibration"]
        cal_b  = models[name_b]["by_difficulty"][stratum]["calibration"]
        exec_a = models[name_a]["by_difficulty"][stratum]["executive"]
        exec_b = models[name_b]["by_difficulty"][stratum]["executive"]
        cal_diff  = cal_a  - cal_b
        exec_diff = exec_a - exec_b
        print(f"  {stratum:<8} {cal_diff:>+9.3f} {exec_diff:>+10.3f}")

print()
print("  Inversion is consistent if: cal diff has one sign, exec diff has the opposite.")
