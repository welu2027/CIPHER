"""Dataset diagnostics: prompt lengths, trigger/effect frequencies, hidden-rule
count distribution, challenge rate, and impact analysis.

Runs entirely without any LLM API. Requires only the pre-generated JSONL and
the cipher Python library.

Usage:
    python analysis/dataset_diagnostics.py [--data data/instances.jsonl] [--beam 8] [--sample N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import statistics
from collections import Counter, defaultdict
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cipher.world import World, State, EntityState, Rule, Trigger, Effect
from cipher.optimal import oracle_score

DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "instances.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rehydrate_world(rec: Dict[str, Any]):
    hidden = rec["hidden"]
    rules = []
    for rr in hidden["rules"]:
        t = rr["trigger"]
        e = rr["effect"]
        trigger = Trigger(kind=t["kind"], i=t["i"], j=t.get("j", -1), k=t.get("k", 0))
        effect = Effect(kind=e["kind"], target=e["target"],
                        delta=e.get("delta", 0), source=e.get("source", -1))
        rules.append(Rule(name=rr["name"], trigger=trigger, effect=effect))
    initial = State(tuple(
        EntityState(phase=e["phase"], flux=e["flux"])
        for e in hidden["initial_state"]
    ))
    world = World(initial=initial, rules=tuple(rules), horizon=hidden["horizon"])
    return world, hidden["hidden_rule_indices"]


# ---------------------------------------------------------------------------
# Section 1: Prompt lengths
# ---------------------------------------------------------------------------

def prompt_length_stats(records: List[Dict]) -> Dict:
    lengths = [len(r["prompt"]) for r in records]
    by_diff: Dict[str, List[int]] = defaultdict(list)
    for r in records:
        by_diff[r["difficulty"]].append(len(r["prompt"]))
    return {"all": lengths, "by_difficulty": dict(by_diff)}


# ---------------------------------------------------------------------------
# Section 2: Trigger and effect kind frequencies
# ---------------------------------------------------------------------------

def trigger_effect_frequencies(records: List[Dict]):
    trigger_counts: Counter = Counter()
    effect_counts: Counter = Counter()
    for r in records:
        for rule in r["hidden"]["rules"]:
            trigger_counts[rule["trigger"]["kind"]] += 1
            effect_counts[rule["effect"]["kind"]] += 1
    return trigger_counts, effect_counts


# ---------------------------------------------------------------------------
# Section 3: Hidden-rule count distribution
# ---------------------------------------------------------------------------

def hidden_count_distribution(records: List[Dict]):
    dist: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        diff = r["difficulty"]
        n_hidden = len(r["hidden"]["hidden_rule_indices"])
        dist[diff][n_hidden] += 1
    return dict(dist)


# ---------------------------------------------------------------------------
# Section 4 & 5: Challenge rate, impact distribution, better-contingency rate
# ---------------------------------------------------------------------------

def compute_simulation_stats(records: List[Dict], beam_width: int = 8):
    """
    For each instance (or sample):
      - challenge_flag: any hidden rule has non-zero impact on oracle trajectory
      - impact values: |oracle_obj - reduced_world_obj| for each hidden rule
      - better_contingency_flag: adversarial-world oracle outperforms the
        full-world oracle plan when run on the adversarial world
    """
    challenge_flags: List[bool] = []
    impact_values: List[float] = []
    near_zero_impacts: int = 0
    material_impacts: int = 0
    better_contingency_flags: List[bool] = []

    total = len(records)
    for idx, rec in enumerate(records):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1}/{total}]", end="\r", flush=True)

        world, hidden_indices = rehydrate_world(rec)
        rules = list(world.rules)

        # Oracle plan on full world (stored oracle_best for obj, but we need
        # the actual plan for ablation, so we re-run beam search)
        best_obj, oracle_plan = oracle_score(world, beam_width=beam_width)

        # Per-hidden-rule ablation
        instance_impacts: List[float] = []
        for rule_idx in hidden_indices:
            reduced_rules = tuple(r for i, r in enumerate(rules) if i != rule_idx)
            reduced_world = World(initial=world.initial,
                                  rules=reduced_rules, horizon=world.horizon)
            reduced_obj = reduced_world.objective(reduced_world.execute(oracle_plan))
            imp = abs(best_obj - reduced_obj)
            instance_impacts.append(imp)
            impact_values.append(float(imp))
            if imp == 0:
                near_zero_impacts += 1
            else:
                material_impacts += 1

        has_challenge = any(imp > 0 for imp in instance_impacts)
        challenge_flags.append(has_challenge)

        # Adversarial world: replace each hidden rule's effect with zero_flux
        adv_rules = list(rules)
        for rule_idx in hidden_indices:
            orig = rules[rule_idx]
            adv_effect = Effect(kind="zero_flux", target=orig.effect.target,
                                delta=0, source=orig.effect.source)
            adv_rules[rule_idx] = Rule(name=orig.name, trigger=orig.trigger,
                                       effect=adv_effect)
        adv_world = World(initial=world.initial,
                          rules=tuple(adv_rules), horizon=world.horizon)

        # Best achievable on adversarial world
        adv_best_obj, _ = oracle_score(adv_world, beam_width=beam_width)
        # Full-world oracle plan applied to adversarial world
        full_plan_on_adv = adv_world.objective(adv_world.execute(oracle_plan))
        better_contingency_flags.append(adv_best_obj > full_plan_on_adv)

    print()  # newline after progress
    return {
        "challenge_flags": challenge_flags,
        "impact_values": impact_values,
        "near_zero_impacts": near_zero_impacts,
        "material_impacts": material_impacts,
        "better_contingency_flags": better_contingency_flags,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def pct(n, total):
    return f"{100*n/total:.1f}%" if total else "N/A"


def fmt_stats(vals):
    if not vals:
        return "N/A"
    return (f"min={min(vals):.0f}  median={statistics.median(vals):.0f}"
            f"  mean={statistics.mean(vals):.0f}  max={max(vals):.0f}")


def report(records, beam_width, sim_stats):
    n = len(records)
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  CIPHER DATASET DIAGNOSTICS  (n={n})")
    print(sep)

    # --- Section 1: Prompt lengths ---
    pls = prompt_length_stats(records)
    print("\n--- 1. PROMPT LENGTH (characters) ---")
    print(f"  All:    {fmt_stats(pls['all'])}")
    for diff in ["easy", "medium", "hard"]:
        vals = pls["by_difficulty"].get(diff, [])
        print(f"  {diff.capitalize():6}: {fmt_stats(vals)}  (n={len(vals)})")

    # --- Section 2: Trigger / effect frequencies ---
    tc, ec = trigger_effect_frequencies(records)
    total_rules = sum(tc.values())
    print("\n--- 2. TRIGGER KIND FREQUENCIES ---")
    for kind, cnt in sorted(tc.items(), key=lambda x: -x[1]):
        print(f"  {kind:<20} {cnt:>5}  ({pct(cnt, total_rules)})")

    print("\n--- 3. EFFECT KIND FREQUENCIES ---")
    for kind, cnt in sorted(ec.items(), key=lambda x: -x[1]):
        print(f"  {kind:<20} {cnt:>5}  ({pct(cnt, total_rules)})")

    # --- Section 3: Hidden-rule count distribution ---
    hc = hidden_count_distribution(records)
    print("\n--- 4. HIDDEN-RULE COUNT DISTRIBUTION ---")
    for diff in ["easy", "medium", "hard"]:
        dist = hc.get(diff, Counter())
        total_diff = sum(dist.values())
        parts = ", ".join(f"{k} hidden: {v} ({pct(v, total_diff)})"
                          for k, v in sorted(dist.items()))
        print(f"  {diff.capitalize():6}: {parts}")

    # --- Section 4: Challenge rate and impact ---
    if sim_stats is None:
        print("\n--- 5-7. SIMULATION STATS: skipped (--skip-sim) ---")
        return

    challenge_flags = sim_stats["challenge_flags"]
    impact_values = sim_stats["impact_values"]
    near_zero = sim_stats["near_zero_impacts"]
    material = sim_stats["material_impacts"]
    better_cont = sim_stats["better_contingency_flags"]

    n_challenged = sum(challenge_flags)
    total_hidden_rules = near_zero + material

    print("\n--- 5. CHALLENGE RATE (hidden rules change oracle trajectory) ---")
    print(f"  Instances where ≥1 hidden rule has non-zero impact: "
          f"{n_challenged}/{n} ({pct(n_challenged, n)})")
    print(f"  Instances where all hidden rules have zero impact:  "
          f"{n - n_challenged}/{n} ({pct(n - n_challenged, n)})")

    print("\n--- 6. HIDDEN-RULE IMPACT DISTRIBUTION ---")
    print(f"  Total hidden-rule slots evaluated: {total_hidden_rules}")
    print(f"  Near-zero impact (delta=0):        {near_zero} ({pct(near_zero, total_hidden_rules)})")
    print(f"  Material impact  (delta>0):        {material} ({pct(material, total_hidden_rules)})")
    if impact_values:
        nonzero = [v for v in impact_values if v > 0]
        print(f"  Impact (all):    {fmt_stats(impact_values)}")
        if nonzero:
            print(f"  Impact (>0 only): {fmt_stats(nonzero)}")

    print("\n--- 7. BETTER-CONTINGENCY RATE ---")
    n_better = sum(better_cont)
    print(f"  Instances where adversarial-world oracle > oracle-plan-on-adv-world:")
    print(f"  {n_better}/{n} ({pct(n_better, n)})")
    print(f"  (= fraction where a better contingency plan exists than the oracle's primary plan)")

    # By difficulty
    by_diff_challenge: Dict[str, List] = defaultdict(list)
    by_diff_better: Dict[str, List] = defaultdict(list)
    for rec, cf, bc in zip(records, challenge_flags, better_cont):
        by_diff_challenge[rec["difficulty"]].append(cf)
        by_diff_better[rec["difficulty"]].append(bc)
    print("\n  By difficulty:")
    for diff in ["easy", "medium", "hard"]:
        cf_list = by_diff_challenge[diff]
        bc_list = by_diff_better[diff]
        if not cf_list:
            continue
        n_d = len(cf_list)
        print(f"    {diff.capitalize():6}: challenge={sum(cf_list)}/{n_d} "
              f"({pct(sum(cf_list), n_d)})  "
              f"better-contingency={sum(bc_list)}/{n_d} "
              f"({pct(sum(bc_list), n_d)})")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--beam", type=int, default=8,
                    help="Beam width for oracle computation (default 8; use 16+ for accuracy)")
    ap.add_argument("--sample", type=int, default=None,
                    help="Use only the first N records (for quick testing)")
    ap.add_argument("--skip-sim", action="store_true",
                    help="Skip simulation-based sections (much faster)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "dataset_diagnostics.txt"),
                    help="Write report to this text file (default: analysis/dataset_diagnostics.txt)")
    args = ap.parse_args()

    print(f"Loading {args.data}...")
    records = load_records(args.data)
    if args.sample:
        records = records[:args.sample]
    print(f"Loaded {len(records)} records.")

    if not args.skip_sim:
        print(f"Running simulation stats (beam_width={args.beam})...")
        sim_stats = compute_simulation_stats(records, beam_width=args.beam)
    else:
        sim_stats = None

    # Capture report output and write to file + stdout
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    report(records, args.beam, sim_stats)
    sys.stdout = old_stdout
    output = buf.getvalue()
    print(output)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
