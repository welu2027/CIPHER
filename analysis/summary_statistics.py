"""Dataset summary statistics table for CIPHER paper.

Produces a single compact table covering:
  - Instance counts by difficulty
  - Entity count, rule count, visible/hidden rule counts (mean ± std)
  - Horizon (mean ± std)
  - Oracle objective best/worst/span (mean ± std)
  - Prompt length in tokens (approx, chars/4) (mean ± std)
  - Trigger and effect kind frequencies across all rules
  - Hidden-rule count distribution per difficulty
  - Fraction of metacog components that are truly unknown

Runs entirely without any LLM API.

Usage:
    python analysis/summary_statistics.py [--data data/instances.jsonl]
        [--out results/reports/summary_statistics.txt]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "instances.jsonl")
DEFAULT_OUT  = os.path.join(os.path.dirname(__file__), "..", "results", "reports", "summary_statistics.txt")

DIFFICULTIES = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ms(vals: List[float], fmt: str = ".2f") -> str:
    """Format mean ± std."""
    if not vals:
        return "—"
    mu = statistics.mean(vals)
    if len(vals) < 2:
        return f"{mu:{fmt}}"
    sd = statistics.stdev(vals)
    return f"{mu:{fmt}} ± {sd:{fmt}}"


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


def load(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Per-record feature extraction
# ---------------------------------------------------------------------------

def extract(rec: Dict[str, Any]) -> Dict[str, Any]:
    h = rec["hidden"]
    n_rules   = len(h["rules"])
    n_visible = len(h["visible_rule_indices"])
    n_hidden  = len(h["hidden_rule_indices"])
    n_ents    = len(h["initial_state"])
    horizon   = h["horizon"]
    oracle_b  = h.get("oracle_best")
    oracle_w  = h.get("oracle_worst")
    span      = (oracle_b - oracle_w) if (oracle_b is not None and oracle_w is not None) else None
    prompt_chars = len(rec.get("prompt", ""))
    approx_tok   = prompt_chars / 4.0

    trigger_kinds = [r["trigger"]["kind"] for r in h["rules"]]
    effect_kinds  = [r["effect"]["kind"]  for r in h["rules"]]

    # Fraction of metacog components that are truly unknown
    mc = h.get("metacog_ground_truth", [])
    n_unknown = sum(1 for x in mc if not x["true_known"])
    frac_unknown = n_unknown / len(mc) if mc else 0.0

    return {
        "difficulty": rec["difficulty"],
        "n_entities": n_ents,
        "n_rules":    n_rules,
        "n_visible":  n_visible,
        "n_hidden":   n_hidden,
        "horizon":    horizon,
        "oracle_best":  oracle_b,
        "oracle_worst": oracle_w,
        "oracle_span":  span,
        "prompt_chars": prompt_chars,
        "approx_tok":   approx_tok,
        "trigger_kinds": trigger_kinds,
        "effect_kinds":  effect_kinds,
        "frac_unknown":  frac_unknown,
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(records: List[Dict[str, Any]]) -> str:
    feats = [extract(r) for r in records]
    by_diff: Dict[str, List] = defaultdict(list)
    for f in feats:
        by_diff[f["difficulty"]].append(f)

    lines: List[str] = []
    sep  = "=" * 72
    sep2 = "-" * 72

    lines += [sep,
              "  CIPHER DATASET — SUMMARY STATISTICS",
              f"  Total instances: {len(records)}",
              sep]

    # -----------------------------------------------------------------------
    # Table 1: Per-difficulty overview
    # -----------------------------------------------------------------------
    lines += ["", "TABLE 1: INSTANCE COUNTS AND STRUCTURAL PROPERTIES", sep2]
    col_w = 14
    header = (f"  {'Statistic':<26}"
              + "".join(f"{'Easy':>{col_w}}{'Medium':>{col_w}}{'Hard':>{col_w}}")
              + f"{'Overall':>{col_w}}")
    lines.append(header)
    lines.append("  " + "-" * (26 + col_w * 4))

    def row(label: str, key: str, fmt: str = ".2f"):
        parts = []
        all_vals = []
        for d in DIFFICULTIES:
            vals = [f[key] for f in by_diff[d] if f[key] is not None]
            parts.append(ms(vals, fmt))
            all_vals.extend(vals)
        parts.append(ms(all_vals, fmt))
        line = f"  {label:<26}" + "".join(f"{p:>{col_w}}" for p in parts)
        lines.append(line)

    row("N instances",       "n_entities")   # placeholder — use counts directly

    # Override first row with raw counts
    lines.pop()
    count_parts = [str(len(by_diff[d])) for d in DIFFICULTIES] + [str(len(feats))]
    lines.append(f"  {'N instances':<26}" + "".join(f"{p:>{col_w}}" for p in count_parts))

    row("Entities (n)",      "n_entities",  ".2f")
    row("Rules total",       "n_rules",     ".2f")
    row("Visible rules",     "n_visible",   ".2f")
    row("Hidden rules",      "n_hidden",    ".2f")
    row("Horizon",           "horizon",     ".2f")
    row("Oracle best",       "oracle_best", ".2f")
    row("Oracle worst",      "oracle_worst",".2f")
    row("Oracle span",       "oracle_span", ".2f")
    row("Prompt tokens*",    "approx_tok",  ".0f")
    row("Frac. unknown MC†", "frac_unknown",".3f")

    lines += ["",
              "  * Approximate token count (chars / 4).",
              "  † Fraction of (rule, component) pairs in metacog_ground_truth",
              "    where true_known = False (averaged per instance)."]

    # -----------------------------------------------------------------------
    # Table 2: Hidden-rule count distribution
    # -----------------------------------------------------------------------
    lines += ["", "", "TABLE 2: HIDDEN-RULE COUNT DISTRIBUTION", sep2]
    all_hidden_counts = Counter(f["n_hidden"] for f in feats)
    max_hidden = max(all_hidden_counts)
    header2 = f"  {'# Hidden rules':<18}" + "".join(f"{'Easy':>10}{'Medium':>10}{'Hard':>10}") + f"{'Total':>10}"
    lines.append(header2)
    lines.append("  " + "-" * (18 + 40))
    for k in range(0, max_hidden + 1):
        parts = []
        total_k = 0
        for d in DIFFICULTIES:
            c = Counter(f["n_hidden"] for f in by_diff[d])[k]
            n_d = len(by_diff[d])
            parts.append(f"{c} ({pct(c, n_d)})")
            total_k += c
        parts.append(f"{total_k} ({pct(total_k, len(feats))})")
        lines.append(f"  {k:<18}" + "".join(f"{p:>10}" for p in parts))

    # -----------------------------------------------------------------------
    # Table 3: Trigger kind frequencies
    # -----------------------------------------------------------------------
    all_trigger: Counter = Counter()
    trigger_by_diff: Dict[str, Counter] = defaultdict(Counter)
    for f in feats:
        for k in f["trigger_kinds"]:
            all_trigger[k] += 1
            trigger_by_diff[f["difficulty"]][k] += 1

    total_rules_all = sum(all_trigger.values())

    lines += ["", "", "TABLE 3: TRIGGER KIND FREQUENCIES (fraction of all rules)", sep2]
    header3 = f"  {'Trigger kind':<22}" + "".join(f"{'Easy':>10}{'Medium':>10}{'Hard':>10}") + f"{'Overall':>10}"
    lines.append(header3)
    lines.append("  " + "-" * (22 + 40))
    for kind in sorted(all_trigger, key=lambda x: -all_trigger[x]):
        parts = []
        for d in DIFFICULTIES:
            c = trigger_by_diff[d][kind]
            total_d = sum(trigger_by_diff[d].values())
            parts.append(f"{pct(c, total_d)}")
        parts.append(f"{pct(all_trigger[kind], total_rules_all)}")
        lines.append(f"  {kind:<22}" + "".join(f"{p:>10}" for p in parts))

    # -----------------------------------------------------------------------
    # Table 4: Effect kind frequencies
    # -----------------------------------------------------------------------
    all_effect: Counter = Counter()
    effect_by_diff: Dict[str, Counter] = defaultdict(Counter)
    for f in feats:
        for k in f["effect_kinds"]:
            all_effect[k] += 1
            effect_by_diff[f["difficulty"]][k] += 1

    total_eff_all = sum(all_effect.values())

    lines += ["", "", "TABLE 4: EFFECT KIND FREQUENCIES (fraction of all rules)", sep2]
    header4 = f"  {'Effect kind':<22}" + "".join(f"{'Easy':>10}{'Medium':>10}{'Hard':>10}") + f"{'Overall':>10}"
    lines.append(header4)
    lines.append("  " + "-" * (22 + 40))
    for kind in sorted(all_effect, key=lambda x: -all_effect[x]):
        parts = []
        for d in DIFFICULTIES:
            c = effect_by_diff[d][kind]
            total_d = sum(effect_by_diff[d].values())
            parts.append(f"{pct(c, total_d)}")
        parts.append(f"{pct(all_effect[kind], total_eff_all)}")
        lines.append(f"  {kind:<22}" + "".join(f"{p:>10}" for p in parts))

    lines += ["", sep, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out",  default=DEFAULT_OUT)
    args = ap.parse_args()

    print(f"Loading {args.data}...")
    records = load(args.data)
    print(f"Loaded {len(records)} records.")

    report = build_report(records)
    print(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
