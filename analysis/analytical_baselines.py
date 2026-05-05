"""
Analytical baselines — no LLM inference required.

1. Calibration bounds
   - Perfect calibration : claim exactly correct known/unknown with confidence=1.
     Brier = 0 -> calibration = 1.0 (derived analytically per difficulty).
   - All-known           : claim known=True, confidence=1 for every component.
   - All-unknown         : claim known=False, confidence=1 for every component.

2. Objective floor
   - No-op   : execute [wait]*horizon, normalize with oracle best/worst.
   - Random  : uniform-random action per step, averaged over N_RANDOM seeds.
   Normalization values (best_objective, worst_objective) are taken from a
   reference model JSON that has all 1000 instances intact (gpt_54_mini).

3. n_hidden ablation
   Already encoded in difficulty: easy=1 hidden, medium=2, hard=3.
   Just reads results.json by_difficulty and formats it.
"""

import json, os, random, sys
from cipher.generator import generate_instance
from cipher.world import Action, all_actions

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, "analytical_baselines.txt")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")

INSTANCES_PATHS = [
    os.path.join(SCRIPT_DIR, "../data/instances.jsonl"),
    os.path.join(SCRIPT_DIR, "../../data/instances.jsonl"),
]
instances_path = next(p for p in INSTANCES_PATHS if os.path.exists(p))
with open(instances_path) as f:
    INSTANCES = [json.loads(l) for l in f if l.strip()]

# --------------------------------------------------------------------------
# 1. Calibration baselines (analytical — no simulation)
# --------------------------------------------------------------------------
# Components per rule: trigger_kind, trigger_k, effect_kind, effect_delta = 4
COMPONENTS = 4
DIFF_PARAMS = {
    "easy":   {"n_rules": 4, "n_hidden": 1},   # 3 visible + 1 hidden
    "medium": {"n_rules": 5, "n_hidden": 2},   # 3 visible + 2 hidden
    "hard":   {"n_rules": 6, "n_hidden": 3},   # 3 visible + 3 hidden
}

def calib_from_brier(brier):
    return max(0.0, 1.0 - brier)

calib_rows = []
for diff, p in DIFF_PARAMS.items():
    n_total   = p["n_rules"] * COMPONENTS
    n_hidden  = p["n_hidden"] * COMPONENTS
    n_visible = (p["n_rules"] - p["n_hidden"]) * COMPONENTS

    # Perfect: Brier = 0 everywhere
    perfect = 1.0

    # All-known: all hidden components scored as p_known=1 against truth=0
    brier_all_known = n_hidden / n_total
    all_known = calib_from_brier(brier_all_known)

    # All-unknown: all visible components scored as p_known=0 against truth=1
    brier_all_unknown = n_visible / n_total
    all_unknown = calib_from_brier(brier_all_unknown)

    calib_rows.append((diff, perfect, all_known, all_unknown, n_total, n_hidden))

# --------------------------------------------------------------------------
# 2. Objective floor (simulation)
# --------------------------------------------------------------------------
# Use gpt_54_mini as normalization source (n=1000, no failures after cleaning)
REF_MODEL = "gpt_54_mini.json"
ref_path  = os.path.join(MODELS_DIR, REF_MODEL)
if not os.path.exists(ref_path):
    print(f"ERROR: reference model file not found: {ref_path}", file=sys.stderr)
    sys.exit(1)

with open(ref_path) as f:
    ref_data = json.load(f)
ref_subruns = ref_data["subruns"]

if len(ref_subruns) != 1000:
    print(f"WARNING: reference model has {len(ref_subruns)} subruns, expected 1000.")
    print("Normalization will only cover available instances.")

N_RANDOM = 20   # random plans per instance

def norm_obj(raw, best, worst):
    span = best - worst
    if span <= 0:
        return 1.0
    return max(0.0, min(1.0, (raw - worst) / span))

noop_by_diff   = {"easy": [], "medium": [], "hard": []}
random_by_diff = {"easy": [], "medium": [], "hard": []}

print("Computing objective baselines (1000 instances)...")
for i, (inst_meta, subrun) in enumerate(zip(INSTANCES, ref_subruns)):
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/1000...", flush=True)

    dr   = subrun["results"][0]["dictResult"]
    best  = int(dr["best_objective"])
    worst = int(dr["worst_objective"])
    diff  = inst_meta["difficulty"]

    inst  = generate_instance(inst_meta["seed"], diff)
    world = inst.world
    h     = world.horizon

    # --- No-op ---
    noop_state = world.execute([Action("wait")] * h)
    noop_score = norm_obj(world.objective(noop_state), best, worst)
    noop_by_diff[diff].append(noop_score)

    # --- Random ---
    actions = all_actions(world.initial.n)
    rand_vals = []
    for rseed in range(N_RANDOM):
        rng  = random.Random(rseed * 7919 + i)
        plan = [rng.choice(actions) for _ in range(h)]
        rand_vals.append(norm_obj(world.objective(world.execute(plan)), best, worst))
    random_by_diff[diff].append(sum(rand_vals) / len(rand_vals))

print("Done.\n")

def mean(xs): return sum(xs) / len(xs) if xs else 0.0

# --------------------------------------------------------------------------
# 3. n_hidden ablation — read results.json
# --------------------------------------------------------------------------
with open(RESULTS_PATH) as f:
    results = json.load(f)

DIMS = ["composite", "objective", "calibration", "attention", "executive"]

# --------------------------------------------------------------------------
# Write output
# --------------------------------------------------------------------------
lines = []
def W(*args): lines.append(" ".join(str(a) for a in args))
def HR(n=76): lines.append("-" * n)
def HDR(title): HR(); W(title); HR()

HDR("CIPHER ANALYTICAL BASELINES")
W()

# ---- Section A: Calibration bounds ----
W("A. CALIBRATION SCORE BOUNDS (analytical — no simulation)")
W()
W(f"{'Difficulty':<10}  {'n_comps':>7}  {'n_hidden':>8}  {'Perfect':>8}  {'All-known':>10}  {'All-unknown':>12}")
HR()
for diff, perfect, all_known, all_unknown, n_total, n_hidden in calib_rows:
    W(f"  {diff:<8}  {n_total:>7}  {n_hidden:>8}  {perfect:>8.4f}  {all_known:>10.4f}  {all_unknown:>12.4f}")
W()
W("Notes:")
W("  Perfect calibration : claim known/unknown exactly right at confidence=1.0")
W("                        Brier=0 => calibration=1.000 for all difficulties.")
W("  All-known           : overconfidence — claims every component is known.")
W("                        Penalised for hidden components. Score falls with n_hidden.")
W("  All-unknown         : underconfidence — claims nothing is known.")
W("                        Penalised for visible components. Score rises with n_hidden.")
W()

# ---- Section B: Objective floor ----
W("B. OBJECTIVE SCORE FLOOR (simulation over 1000 instances)")
W()
W(f"{'Strategy':<14}  {'easy':>8}  {'medium':>8}  {'hard':>8}  {'overall':>8}  n")
HR()

all_noop   = noop_by_diff["easy"] + noop_by_diff["medium"] + noop_by_diff["hard"]
all_random = random_by_diff["easy"] + random_by_diff["medium"] + random_by_diff["hard"]

W(f"  {'No-op':<12}  "
  f"{mean(noop_by_diff['easy']):>8.4f}  "
  f"{mean(noop_by_diff['medium']):>8.4f}  "
  f"{mean(noop_by_diff['hard']):>8.4f}  "
  f"{mean(all_noop):>8.4f}  {len(all_noop)}")
W(f"  {'Random':<12}  "
  f"{mean(random_by_diff['easy']):>8.4f}  "
  f"{mean(random_by_diff['medium']):>8.4f}  "
  f"{mean(random_by_diff['hard']):>8.4f}  "
  f"{mean(all_random):>8.4f}  {len(all_random)}")
W()
W(f"  Normalization: (raw - worst_obj) / (best_obj - worst_obj)")
W(f"  No-op    = [wait]*7 (rules still fire each step, agent takes no useful action)")
W(f"  Random   = uniform-random from all_actions per step, mean over {N_RANDOM} seeds")
W()

# ---- Section C: n_hidden ablation ----
W("C. n_HIDDEN ABLATION (existing results by difficulty)")
W()
W("  Difficulty encodes n_hidden: easy=1, medium=2, hard=3")
W("  Co-variates: n_entities (3 easy / 4 med,hard), n_rules (4/5/6)")
W()

# Build a table: model vs. objective by difficulty
header = f"  {'Model':<28}  {'easy obj':>8}  {'med obj':>8}  {'hard obj':>8}  {'delta e-h':>10}"
W(header)
HR()
for m in results["models"]:
    bd = m.get("by_difficulty", {})
    e_obj = bd.get("easy",   {}).get("objective", float("nan"))
    m_obj = bd.get("medium", {}).get("objective", float("nan"))
    h_obj = bd.get("hard",   {}).get("objective", float("nan"))
    delta = e_obj - h_obj if (e_obj == e_obj and h_obj == h_obj) else float("nan")
    W(f"  {m['name']:<28}  {e_obj:>8.4f}  {m_obj:>8.4f}  {h_obj:>8.4f}  {delta:>10.4f}")
W()
W("  delta e-h: easy objective minus hard objective.")
W("  A large positive value means the model degrades more when rules are hidden.")
W()

# ---- Summary interpretation ----
HDR("INTERPRETATION NOTES")
W()
W("1. Calibration lower/upper bounds:")
W("   - Models scoring below all-known have LESS overconfidence than a naive oracle.")
W("   - Models scoring near 'perfect' (1.0) have excellent uncertainty quantification.")
W("   - The all-known/all-unknown gap narrows as n_hidden increases:")
W("     easy gap: {:.3f},  medium gap: {:.3f},  hard gap: {:.3f}".format(
      calib_rows[0][2] - calib_rows[0][3],
      calib_rows[1][2] - calib_rows[1][3],
      calib_rows[2][2] - calib_rows[2][3],
  ))
W()
W("2. Objective floor:")
W(f"   No-op    = {mean(all_noop):.4f}  (rules fire but agent contributes nothing)")
W(f"   Random   = {mean(all_random):.4f}  (random actions, averaged over {N_RANDOM} seeds)")
W("   Any model scoring at or near these values is not planning effectively.")
W()
W("3. n_hidden ablation:")
W("   Objective typically falls with difficulty. If a model's easy-hard delta is")
W("   near zero, it is either insensitive to uncertainty or saturation-floored.")
W()

output = "\n".join(lines)
print(output)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(output + "\n")
print(f"\nWrote {OUTPUT_PATH}")
