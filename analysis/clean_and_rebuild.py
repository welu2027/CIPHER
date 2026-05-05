"""
Remove _ZERO / parse-error subruns from analysis/models/*.json in place,
embedding the difficulty label from instances.jsonl into each surviving
subrun before removal so positional alignment is no longer needed.
Then rebuilds analysis/results.json with valid-only aggregate scores.

A subrun is a failure if its dictResult lacks 'best_objective' or has
parse_errors > 0.
"""

import json, os

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")

INSTANCES_PATHS = [
    os.path.join(SCRIPT_DIR, "../data/instances.jsonl"),
    os.path.join(SCRIPT_DIR, "../../data/instances.jsonl"),
]
instances_path = next(p for p in INSTANCES_PATHS if os.path.exists(p))
with open(instances_path) as f:
    INSTANCES = [json.loads(l) for l in f if l.strip()]

DIFFICULTIES = [inst["difficulty"] for inst in INSTANCES]   # 1000-length list

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

DIMS = ["composite", "objective", "calibration", "attention", "executive"]

def is_failure(dr):
    return "best_objective" not in dr or float(dr.get("parse_errors", 0)) > 0

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

results_models = []

for name, fname in FILE_MAP.items():
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        print(f"  SKIP {name:<28} — file missing")
        continue
    with open(path) as f:
        d = json.load(f)
    if not d or "subruns" not in d:
        print(f"  SKIP {name:<28} — no data")
        continue

    before = len(d["subruns"])
    assert before == len(DIFFICULTIES), \
        f"{name}: expected {len(DIFFICULTIES)} subruns, got {before}"

    # Embed difficulty into each subrun's dictResult before filtering
    for i, sub in enumerate(d["subruns"]):
        sub["results"][0]["dictResult"]["difficulty"] = DIFFICULTIES[i]

    # Remove failures
    d["subruns"] = [
        s for s in d["subruns"]
        if not is_failure(s["results"][0]["dictResult"])
    ]
    after = len(d["subruns"])

    with open(path, "w") as f:
        json.dump(d, f)

    print(f"  {name:<28}  {before} -> {after}  (removed {before - after})")

    # Aggregate for results.json
    rows  = [s["results"][0]["dictResult"] for s in d["subruns"]]
    diffs = [r["difficulty"] for r in rows]

    def agg(subset):
        n = len(subset)
        return {dim: round(mean([float(r.get(dim, 0)) for r in subset]), 6)
                for dim in DIMS}, n

    overall, n_total = agg(rows)

    by_difficulty = {}
    for stratum in ["easy", "medium", "hard"]:
        subset = [r for r, diff in zip(rows, diffs) if diff == stratum]
        scores, n_s = agg(subset)
        by_difficulty[stratum] = {"n": n_s, **scores}

    results_models.append({
        "name":         name,
        "n":            n_total,
        "overall":      overall,
        "by_difficulty": by_difficulty,
    })

with open(RESULTS_PATH, "w") as f:
    json.dump({"models": results_models}, f, indent=2)

print(f"\nWrote {RESULTS_PATH} with {len(results_models)} models.")
print("Difficulty labels embedded in each subrun — offline_analysis.py will")
print("need a small update to read 'difficulty' from dictResult instead of")
print("instances.jsonl. Run update_loader.py or update manually.")
