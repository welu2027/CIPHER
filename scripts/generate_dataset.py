"""Generate a JSONL dataset of CIPHER instances.

Each line is a JSON object with:
  id, seed, difficulty
  prompt                  <- the text shown to the model
  hidden                  <- ground-truth fields (do NOT show to model)
    rules: full rule semantics
    metacog_ground_truth
    true_unknown_ranking
    initial_state
    horizon
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random

# allow running as `python scripts/generate_dataset.py` from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cipher import generate_instance, build_prompt
from cipher.optimal import oracle_score
from cipher.scorer import _worst_objective


DIFFICULTY_MIX = [("easy", 0.25), ("medium", 0.5), ("hard", 0.25)]


def _instance_to_record(inst, include_oracle: bool) -> dict:
    rules_full = []
    for r in inst.world.rules:
        rules_full.append({
            "name": r.name,
            "trigger": {"kind": r.trigger.kind, "i": r.trigger.i,
                        "j": r.trigger.j, "k": r.trigger.k},
            "effect": {"kind": r.effect.kind, "target": r.effect.target,
                       "delta": r.effect.delta, "source": r.effect.source},
        })
    rec = {
        "id": inst.id,
        "seed": inst.seed,
        "difficulty": inst.difficulty,
        "prompt": build_prompt(inst),
        "hidden": {
            "rules": rules_full,
            "visible_rule_indices": inst.visible_rule_indices,
            "hidden_rule_indices": inst.hidden_rule_indices,
            "metacog_ground_truth": inst.metacog_ground_truth,
            "true_unknown_ranking": inst.true_unknown_ranking,
            "initial_state": [{"phase": e.phase, "flux": e.flux}
                              for e in inst.world.initial.entities],
            "horizon": inst.world.horizon,
            "hidden_fields": inst.hidden_fields,
        },
    }
    if include_oracle:
        best, _ = oracle_score(inst.world)
        worst = _worst_objective(inst.world)
        rec["hidden"]["oracle_best"] = best
        rec["hidden"]["oracle_worst"] = worst
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="data/instances.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle", action="store_true",
                    help="precompute best/worst objectives (slower)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # pick difficulties according to the mix
    difficulties = []
    for label, frac in DIFFICULTY_MIX:
        difficulties += [label] * max(1, int(frac * args.n))
    while len(difficulties) < args.n:
        difficulties.append("medium")
    difficulties = difficulties[: args.n]
    rng.shuffle(difficulties)

    with open(args.out, "w") as f:
        for i, diff in enumerate(difficulties):
            sub_seed = rng.randrange(2 ** 31)
            inst = generate_instance(seed=sub_seed, difficulty=diff)
            rec = _instance_to_record(inst, include_oracle=args.oracle)
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 25 == 0:
                print(f"generated {i+1}/{args.n}", file=sys.stderr)

    print(f"wrote {args.n} instances to {args.out}")


if __name__ == "__main__":
    main()
