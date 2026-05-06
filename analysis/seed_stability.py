"""Seed-stability analysis for CIPHER.

Generates mini-datasets for 3 additional seeds beyond the canonical seed-2026
and compares structural statistics (trigger/effect distributions, entity counts,
hidden-rule counts, oracle objective statistics).

Runs entirely without any LLM API.

Usage:
    python analysis/seed_stability.py [--n 200] [--beam 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import statistics
import random
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cipher.generator import generate_instance, _diff_params
from cipher.optimal import oracle_score
from cipher.world import World


DIFFICULTY_MIX = [("easy", 0.25), ("medium", 0.50), ("hard", 0.25)]
CANONICAL_SEED = 2026
EXTRA_SEEDS = [2027, 2028, 2029]


def generate_mini_dataset(master_seed: int, n: int, beam_width: int) -> List[Dict]:
    """Generate n instances with the same 25/50/25 easy/medium/hard split."""
    rng = random.Random(master_seed)
    records = []
    counts = {"easy": int(n * 0.25), "medium": int(n * 0.50), "hard": int(n * 0.25)}
    # Handle rounding
    while sum(counts.values()) < n:
        counts["medium"] += 1

    pool = []
    for diff, cnt in counts.items():
        for _ in range(cnt):
            pool.append(diff)
    rng.shuffle(pool)

    instance_seed_base = master_seed * 10_000
    for i, diff in enumerate(pool):
        inst_seed = instance_seed_base + i
        inst = generate_instance(inst_seed, diff)
        best_obj, _ = oracle_score(inst.world, beam_width=beam_width)
        from cipher.scorer import _worst_objective
        worst_obj = _worst_objective(inst.world, beam_width=beam_width)
        records.append({
            "seed": master_seed,
            "inst_seed": inst_seed,
            "difficulty": diff,
            "n_entities": inst.world.initial.n,
            "n_rules": len(inst.world.rules),
            "n_hidden": len(inst.hidden_rule_indices),
            "oracle_best": best_obj,
            "oracle_worst": worst_obj,
            "trigger_kinds": [inst.world.rules[i].trigger.kind
                              for i in range(len(inst.world.rules))],
            "effect_kinds": [inst.world.rules[i].effect.kind
                             for i in range(len(inst.world.rules))],
        })
        if (i + 1) % 20 == 0:
            print(f"  seed={master_seed}: [{i+1}/{n}]", end="\r", flush=True)
    print(f"  seed={master_seed}: done ({n} instances)       ")
    return records


def also_load_canonical(n: int, beam_width: int, canonical_jsonl: str) -> List[Dict] | None:
    """Load first n records from the pre-generated canonical JSONL."""
    if not os.path.exists(canonical_jsonl):
        return None
    from cipher.world import State, EntityState, Rule, Trigger, Effect
    results = []
    with open(canonical_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            hidden = rec["hidden"]
            results.append({
                "seed": CANONICAL_SEED,
                "inst_seed": rec["seed"],
                "difficulty": rec["difficulty"],
                "n_entities": len(hidden["initial_state"]),
                "n_rules": len(hidden["rules"]),
                "n_hidden": len(hidden["hidden_rule_indices"]),
                "oracle_best": hidden.get("oracle_best"),
                "oracle_worst": hidden.get("oracle_worst"),
                "trigger_kinds": [r["trigger"]["kind"] for r in hidden["rules"]],
                "effect_kinds": [r["effect"]["kind"] for r in hidden["rules"]],
            })
            if len(results) >= n:
                break
    return results


def compute_stats(records: List[Dict]) -> Dict:
    trigger_counts: Counter = Counter()
    effect_counts: Counter = Counter()
    for r in records:
        for k in r["trigger_kinds"]:
            trigger_counts[k] += 1
        for k in r["effect_kinds"]:
            effect_counts[k] += 1

    oracle_bests = [r["oracle_best"] for r in records if r["oracle_best"] is not None]
    oracle_worsts = [r["oracle_worst"] for r in records if r["oracle_worst"] is not None]
    spans = [b - w for b, w in zip(oracle_bests, oracle_worsts)
             if b is not None and w is not None]

    n = len(records)
    diff_counts = Counter(r["difficulty"] for r in records)

    return {
        "n": n,
        "diff_counts": diff_counts,
        "trigger_counts": trigger_counts,
        "effect_counts": effect_counts,
        "oracle_best_mean": statistics.mean(oracle_bests) if oracle_bests else None,
        "oracle_best_std": statistics.stdev(oracle_bests) if len(oracle_bests) > 1 else None,
        "oracle_worst_mean": statistics.mean(oracle_worsts) if oracle_worsts else None,
        "oracle_span_mean": statistics.mean(spans) if spans else None,
        "oracle_span_std": statistics.stdev(spans) if len(spans) > 1 else None,
    }


def report_stability(all_stats: Dict[int, Dict]):
    sep = "=" * 70
    seeds = sorted(all_stats.keys())
    print(f"\n{sep}")
    print("  CIPHER SEED-STABILITY ANALYSIS")
    print(sep)

    # Trigger frequencies
    all_trigger_kinds = sorted({k for s in all_stats.values()
                                 for k in s["trigger_counts"]})
    print("\n--- TRIGGER KIND FREQUENCY (fraction of rules) ---")
    print(f"  {'Kind':<22}" + "".join(f"  seed-{s:<6}" for s in seeds))
    for kind in all_trigger_kinds:
        row = f"  {kind:<22}"
        for seed in seeds:
            sc = all_stats[seed]["trigger_counts"]
            total = sum(sc.values())
            frac = sc.get(kind, 0) / total if total else 0
            row += f"  {frac:>8.3f}"
        print(row)

    # Cross-seed std
    trigger_fracs: Dict[str, List[float]] = defaultdict(list)
    for seed in seeds:
        sc = all_stats[seed]["trigger_counts"]
        total = sum(sc.values())
        for kind in all_trigger_kinds:
            trigger_fracs[kind].append(sc.get(kind, 0) / total if total else 0)
    print("  " + "-"*60)
    print(f"  {'Cross-seed std':<22}" + "".join(
        f"  {statistics.stdev(trigger_fracs[k]):>8.4f}" for k in all_trigger_kinds
    ))

    # Effect frequencies
    all_effect_kinds = sorted({k for s in all_stats.values()
                                for k in s["effect_counts"]})
    print("\n--- EFFECT KIND FREQUENCY (fraction of rules) ---")
    print(f"  {'Kind':<22}" + "".join(f"  seed-{s:<6}" for s in seeds))
    for kind in all_effect_kinds:
        row = f"  {kind:<22}"
        for seed in seeds:
            sc = all_stats[seed]["effect_counts"]
            total = sum(sc.values())
            frac = sc.get(kind, 0) / total if total else 0
            row += f"  {frac:>8.3f}"
        print(row)

    effect_fracs: Dict[str, List[float]] = defaultdict(list)
    for seed in seeds:
        sc = all_stats[seed]["effect_counts"]
        total = sum(sc.values())
        for kind in all_effect_kinds:
            effect_fracs[kind].append(sc.get(kind, 0) / total if total else 0)
    print("  " + "-"*60)
    print(f"  {'Cross-seed std':<22}" + "".join(
        f"  {statistics.stdev(effect_fracs[k]):>8.4f}" for k in all_effect_kinds
    ))

    # Oracle statistics
    print("\n--- ORACLE OBJECTIVE STATISTICS ---")
    header = f"  {'Metric':<25}" + "".join(f"  seed-{s:<9}" for s in seeds)
    print(header)

    def row(label, extractor):
        vals = [extractor(all_stats[s]) for s in seeds]
        cross_std = statistics.stdev([v for v in vals if v is not None]) if len([v for v in vals if v is not None]) > 1 else 0
        parts = "".join(f"  {(v if v is not None else float('nan')):>11.3f}" for v in vals)
        print(f"  {label:<25}{parts}  (cross-seed std={cross_std:.3f})")

    row("oracle_best mean", lambda s: s["oracle_best_mean"])
    row("oracle_best std", lambda s: s["oracle_best_std"])
    row("oracle_worst mean", lambda s: s["oracle_worst_mean"])
    row("oracle_span mean", lambda s: s["oracle_span_mean"])
    row("oracle_span std", lambda s: s["oracle_span_std"])

    print(f"\n{sep}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="Number of instances per extra seed (default 200)")
    ap.add_argument("--beam", type=int, default=8,
                    help="Beam width for oracle computation (default 8)")
    ap.add_argument("--canonical-jsonl",
                    default=os.path.join(os.path.dirname(__file__), "..", "data", "instances.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "seed_stability.txt"),
                    help="Write report to this text file (default: analysis/seed_stability.txt)")
    args = ap.parse_args()

    all_stats: Dict[int, Dict] = {}

    # Load canonical seed from pre-computed JSONL
    print(f"Loading canonical seed {CANONICAL_SEED} from JSONL (first {args.n} records)...")
    canon_records = also_load_canonical(args.n, args.beam, args.canonical_jsonl)
    if canon_records:
        all_stats[CANONICAL_SEED] = compute_stats(canon_records)
        print(f"  Loaded {len(canon_records)} records.")
    else:
        print(f"  WARNING: canonical JSONL not found at {args.canonical_jsonl}")

    # Generate extra seeds
    for seed in EXTRA_SEEDS:
        print(f"Generating seed {seed} ({args.n} instances, beam={args.beam})...")
        records = generate_mini_dataset(seed, args.n, args.beam)
        all_stats[seed] = compute_stats(records)

    # Capture report output and write to file + stdout
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    report_stability(all_stats)
    sys.stdout = old_stdout
    output = buf.getvalue()
    print(output)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
