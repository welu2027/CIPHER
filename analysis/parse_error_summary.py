import json, os
from collections import Counter

models_dir = os.path.join(os.path.dirname(__file__), "models")

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

print(f"{'Model':<28}  {'schema_errors=0':>6}  {'=1':>6}  {'=2':>6}  {'=3':>6}  {'4+':>6}  {'_ZERO':>6}  {'total':>6}")
print("-" * 84)

for name, fname in FILE_MAP.items():
    path = os.path.join(models_dir, fname)
    if not os.path.exists(path):
        print(f"  {name:<28}  [missing]")
        continue
    with open(path) as f:
        d = json.load(f)
    if not d or "subruns" not in d:
        print(f"  {name:<28}  [no data]")
        continue

    counts = Counter()
    for sub in d["subruns"]:
        dr = sub["results"][0]["dictResult"]
        if "best_objective" not in dr:
            counts["total_fail"] += 1
        else:
            counts[int(dr.get("parse_errors", 0))] += 1
    total = sum(counts.values())
    ge4   = sum(v for k, v in counts.items() if isinstance(k, int) and k >= 4)
    print(f"  {name:<28}  {counts[0]:>6}  {counts[1]:>6}  {counts[2]:>6}  {counts[3]:>6}  {ge4:>6}  {counts['total_fail']:>6}  {total:>6}")
