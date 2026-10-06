"""Build the instance file for the separate-probe-budget ablation.

Takes a fixed, difficulty-stratified subset of data/instances.jsonl (same ids,
seeds, hidden ground truth, and oracle bounds as the original runs) and swaps
in the separate-budget prompt (cipher/separate_budget.py). Only the budget
wording changes; the rest of each prompt is byte-identical to the original.

    python3 scripts/make_separate_budget_subset.py            # 1/3 subset (334)
    python3 scripts/make_separate_budget_subset.py --frac 1    # all 1,000

Output: data/instances_separate_budget.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cipher.separate_budget import PROBE_CAP, SCORER_VERSION, make_separate_budget_prompt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "instances.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "instances_separate_budget.jsonl"))
    ap.add_argument("--frac", type=float, default=1 / 3)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(args.seed)
    chosen = []
    for diff in ["easy", "medium", "hard"]:
        pool = [r for r in records if r["difficulty"] == diff]
        k = round(len(pool) * args.frac)
        chosen += rng.sample(pool, k) if k < len(pool) else pool
    order = {r["id"]: i for i, r in enumerate(records)}
    chosen.sort(key=lambda r: order[r["id"]])  # keep dataset order

    with open(args.out, "w", encoding="utf-8") as f:
        for r in chosen:
            out = dict(r)
            out["prompt_original"] = r["prompt"]
            out["prompt"] = make_separate_budget_prompt(r["prompt"], horizon=r["hidden"]["horizon"])
            out["condition"] = {"name": "separate_probe_budget", "probe_cap": PROBE_CAP,
                                "scorer_version": SCORER_VERSION}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    counts = {d: sum(r["difficulty"] == d for r in chosen) for d in ["easy", "medium", "hard"]}
    print(f"wrote {len(chosen)} instances to {args.out}  {counts}")


if __name__ == "__main__":
    main()
