"""Generate all main-paper and appendix figures for CIPHER.

All figures are produced offline — no LLM API required.

Data sources:
  results/summary.json                 — 16-model aggregate scores (overall + by difficulty)
  results/baselines/cautious.json      — stub-cautious per-instance scores
  results/baselines/probe_heavy.json   — stub-probe-heavy per-instance scores
  data/instances.jsonl                 — raw instances (oracle_best, oracle_worst pre-stored)

Output (written to results/figures/):
  fig3_heatmap.pdf             — model × dimension heatmap (main results)
  fig4_scatter.pdf             — Objective vs Executive scatter (main claim)
  fig5_intra_family.pdf        — intra-family comparison (GPT and Claude inversions)
  figA_oracle_span.pdf         — oracle-span / challenge-rate distributions (appendix)
  figB_dimension_dists.pdf     — per-instance dimension distributions (appendix)

Usage:
    python analysis/plot_figures.py [--results results/summary.json]
        [--stub-cautious results/baselines/cautious.json]
        [--stub-probe-heavy results/baselines/probe_heavy.json]
        [--data data/instances.jsonl]
        [--out results/figures]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")
BASELINES = os.path.join(RESULTS, "baselines")

DEFAULT_RESULTS     = os.path.join(RESULTS, "summary.json")
DEFAULT_DATA        = os.path.join(ROOT, "data", "instances.jsonl")
DEFAULT_OUT         = os.path.join(RESULTS, "figures")

# All stub/baseline result files (name -> path relative to repo root)
STUB_FILES: Dict[str, str] = {
    "stub-cautious":    os.path.join(BASELINES, "cautious.json"),
    "stub-probe-heavy": os.path.join(BASELINES, "probe_heavy.json"),
    "stub-greedy":      os.path.join(BASELINES, "greedy.json"),
    "stub-noop":        os.path.join(BASELINES, "noop.json"),
    "stub-random":      os.path.join(BASELINES, "random.json"),
}

DIMS = ["composite", "objective", "calibration", "attention", "executive"]
DIM_LABELS = ["Composite", "Objective", "Calibration", "Attention", "Executive"]

# ---------------------------------------------------------------------------
# Family assignments (keyed on substring of model name)
# ---------------------------------------------------------------------------

FAMILY_PREFIXES = [
    ("GPT",      "GPT",      "#4e79a7"),
    ("Gemini",   "Gemini",   "#f28e2b"),
    ("Claude",   "Claude",   "#e15759"),
    ("DeepSeek", "DeepSeek", "#76b7b2"),
    ("Qwen",     "Qwen",     "#59a14f"),
    ("Gemma",    "Gemma",    "#edc948"),
]
STUB_COLOR = "#b07aa1"


def family_of(name: str) -> Tuple[str, str]:
    """Return (family_label, color) for a model name."""
    for prefix, label, color in FAMILY_PREFIXES:
        if name.startswith(prefix):
            return label, color
    return "Other", "#aaaaaa"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["models"]


def load_stub(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_instances(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def stub_to_model_entry(stub_data: Dict, name: str) -> Dict[str, Any]:
    """Convert stub JSON (summary + per_instance) to summary.json model format."""
    s = stub_data["summary"]
    overall = {d: s[f"mean_{d}"] for d in DIMS}
    # Build by_difficulty from per_instance
    by_diff: Dict[str, Dict] = defaultdict(lambda: defaultdict(list))
    for rec in stub_data.get("per_instance", []):
        diff = rec.get("difficulty", "medium")
        for d in DIMS:
            by_diff[diff][d].append(rec[d])
    by_diff_out = {}
    for diff, dim_lists in by_diff.items():
        entry = {"n": len(next(iter(dim_lists.values())))}
        for d in DIMS:
            vals = dim_lists[d]
            entry[d] = float(np.mean(vals)) if vals else 0.0
        by_diff_out[diff] = entry
    return {"name": name, "overall": overall, "by_difficulty": by_diff_out}


# ---------------------------------------------------------------------------
# Figure 3: Heatmap (model × dimension)
# ---------------------------------------------------------------------------

def fig3_heatmap(models: List[Dict], out_path: str):
    names = [m["name"] for m in models]
    matrix = np.array([[m["overall"][d] for d in DIMS] for m in models])

    # Sort by composite descending
    order = np.argsort(matrix[:, 0])[::-1]
    matrix = matrix[order]
    names = [names[i] for i in order]

    # Square overall figure: axes height driven by rows, width matches it
    fig_h = max(6, 0.4 * len(names) + 2.5)
    fig_w = fig_h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    cb = plt.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("Score (0–1)", fontsize=13)
    cb.ax.tick_params(labelsize=12)

    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels(DIM_LABELS, fontsize=14)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=13)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(DIMS)):
            v = matrix[i, j]
            color = "white" if v < 0.55 or v > 0.85 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    # Color y-tick labels by family
    for ytick, name in zip(ax.get_yticklabels(), names):
        _, color = family_of(name)
        ytick.set_color(color)

    # Legend placed below the heatmap, outside the axes
    legend_items = []
    seen = set()
    for m in [models[i] for i in order]:
        fam, col = family_of(m["name"])
        if fam not in seen:
            legend_items.append(mpatches.Patch(color=col, label=fam))
            seen.add(fam)
    ax.legend(handles=legend_items, loc="upper center",
              bbox_to_anchor=(0.5, -0.04), ncol=len(legend_items),
              fontsize=13, framealpha=0.9, borderaxespad=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Objective vs Executive scatter
# ---------------------------------------------------------------------------

def fig4_scatter(models: List[Dict], out_path: str):
    # Per-model label offsets (dx, dy in points). Tuned for the data layout.
    LABEL_OFFSETS: Dict[str, Tuple[float, float]] = {
        "Claude Opus 4.5":         ( -5,   8),
        "Claude Opus 4.6":         (  6,   3),
        "Claude Sonnet 4.6":       (  6,  -9),
        "Claude Opus 4.7":         ( -5, -10),
        "Claude 4.5 Haiku":        (  6,   3),
        "GPT-5.4 Nano":            (  6,   4),
        "GPT-5.4 mini":            (  6,  -9),
        "GPT-5.5":                 (  6,   3),
        "GPT-5.4":                 (  6,  -9),
        "Gemini 2.5 Pro":          (  6,   4),
        "Gemini 2.5 Flash":        (  6,  -9),
        "Gemini 3 Flash Preview":  ( -5, -10),
        "Gemini 3.1 Pro Preview":  (  6,   3),
        "Qwen 3 Next 80B Instruct":(  6,   3),
        "DeepSeek V3.2":           (  6,  -9),
        "Gemma 4 31B":             (  6,   3),
        # stubs — placed to avoid the main LLM cluster
        "stub-cautious":           (-70,   4),
        "stub-probe-heavy":        (  6,   5),
        "stub-greedy":             (  6,   3),
        "stub-noop":               (  6,  -9),
        "stub-random":             (  6,   3),
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    family_seen: set = set()
    for m in models:
        x = m["overall"]["objective"]
        y = m["overall"]["executive"]
        fam, col = family_of(m["name"])
        label = fam if fam not in family_seen else None
        family_seen.add(fam)
        # stubs drawn slightly smaller and more transparent
        is_stub = m["name"].startswith("stub")
        ax.scatter(x, y, c=col, s=55 if is_stub else 80,
                   alpha=0.65 if is_stub else 1.0,
                   marker="D" if is_stub else "o",
                   zorder=3, label=label)
        dx, dy = LABEL_OFFSETS.get(m["name"], (6, 3))
        short = (m["name"]
                 .replace("Preview", "Prev.")
                 .replace("Instruct", "Inst."))
        ax.annotate(short, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.5, color=col,
                    alpha=0.75 if is_stub else 0.9,
                    style="italic" if is_stub else "normal")

    # Regression line — LLMs only (exclude stubs)
    llm_models = [m for m in models if not m["name"].startswith("stub")]
    xs = np.array([m["overall"]["objective"] for m in llm_models])
    ys = np.array([m["overall"]["executive"] for m in llm_models])
    m_coef, b = np.polyfit(xs, ys, 1)
    x_line = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 100)
    ax.plot(x_line, m_coef * x_line + b, "k--", lw=1.2, alpha=0.5,
            label=f"OLS (LLMs only, slope={m_coef:+.2f})")

    ax.set_xlabel("Objective score", fontsize=11)
    ax.set_ylabel("Executive score (contingency quality)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Legend outside plot, bottom center
    ax.legend(fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=5,
              framealpha=0.9, borderaxespad=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Intra-family comparison (GPT and Claude)
# ---------------------------------------------------------------------------

def fig5_intra_family(models: List[Dict], out_path: str):
    gpt_models    = [m for m in models if m["name"].startswith("GPT")]
    claude_models = [m for m in models if m["name"].startswith("Claude")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    show_dims = ["composite", "objective", "calibration", "executive"]
    show_labels = ["Composite", "Objective", "Calibration", "Executive"]
    colors = ["#4e79a7", "#59a14f", "#e15759", "#b07aa1"]

    def dumbbell(ax, family_models, family_color, title):
        names = [m["name"] for m in family_models]
        x = np.arange(len(show_dims))
        y_positions = np.arange(len(names))

        for i, m in enumerate(family_models):
            vals = [m["overall"][d] for d in show_dims]
            ax.plot(vals, [i] * len(vals), "-", color="gray", lw=1, alpha=0.4, zorder=1)
            for j, (v, col) in enumerate(zip(vals, colors)):
                ax.scatter(v, i, c=col, s=60, zorder=3)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xticks(np.arange(0.4, 1.01, 0.1))
        ax.set_xlim(0.38, 1.02)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Score", fontsize=10)

        # Legend inside first panel only
        if ax == axes[0]:
            patches = [mpatches.Patch(color=col, label=lbl)
                       for col, lbl in zip(colors, show_labels)]
            ax.legend(handles=patches, fontsize=8, loc="lower right")

    dumbbell(axes[0], gpt_models,    "#4e79a7", "GPT family")
    dumbbell(axes[1], claude_models, "#e15759", "Claude family")

    fig.suptitle("Intra-Family Score Profiles", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Appendix A: Oracle span / challenge-rate proxy distributions
# ---------------------------------------------------------------------------

def figA_oracle_span(instances: List[Dict], out_path: str):
    diffs = ["easy", "medium", "hard"]
    diff_colors = {"easy": "#59a14f", "medium": "#4e79a7", "hard": "#e15759"}

    spans: Dict[str, List[float]] = defaultdict(list)
    bests: Dict[str, List[float]] = defaultdict(list)
    for rec in instances:
        h = rec["hidden"]
        diff = rec["difficulty"]
        ob = h.get("oracle_best")
        ow = h.get("oracle_worst")
        if ob is not None and ow is not None:
            spans[diff].append(ob - ow)
            bests[diff].append(ob)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: oracle span distribution
    ax = axes[0]
    for diff in diffs:
        ax.hist(spans[diff], bins=20, alpha=0.6, color=diff_colors[diff],
                label=diff.capitalize(), density=True)
    ax.set_xlabel("Oracle span (best − worst objective)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Oracle Objective Span\nby Difficulty", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: oracle best distribution
    ax = axes[1]
    for diff in diffs:
        ax.hist(bests[diff], bins=20, alpha=0.6, color=diff_colors[diff],
                label=diff.capitalize(), density=True)
    ax.set_xlabel("Oracle best objective", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Oracle Best Objective\nby Difficulty", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Summary stats as text boxes
    for ax, data_dict, xlabel in [
        (axes[0], spans, "span"),
        (axes[1], bests, "best"),
    ]:
        summary_lines = []
        for diff in diffs:
            vals = data_dict[diff]
            summary_lines.append(
                f"{diff.capitalize()}: μ={np.mean(vals):.1f}, σ={np.std(vals):.1f}"
            )
        ax.text(0.03, 0.97, "\n".join(summary_lines), transform=ax.transAxes,
                fontsize=7.5, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Oracle Objective Distributions — Benchmark Difficulty Validation",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Appendix B: Per-instance dimension distributions (stub baselines)
# ---------------------------------------------------------------------------

def figB_dimension_dists(stub_cautious: Dict, stub_probe_heavy: Dict, out_path: str):
    dims_show = ["objective", "calibration", "attention", "executive"]
    dim_labels_show = ["Objective", "Calibration", "Attention", "Executive"]

    sc_instances = stub_cautious.get("per_instance", [])
    sp_instances = stub_probe_heavy.get("per_instance", [])

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()

    stub_pairs = [
        ("stub-cautious",    sc_instances, "#b07aa1"),
        ("stub-probe-heavy", sp_instances, "#76b7b2"),
    ]

    for ax, dim, dlabel in zip(axes, dims_show, dim_labels_show):
        for stub_name, recs, color in stub_pairs:
            vals = [r[dim] for r in recs if dim in r]
            ax.hist(vals, bins=30, alpha=0.55, color=color,
                    label=stub_name, density=True)
            ax.axvline(np.mean(vals), color=color, lw=1.8, ls="--", alpha=0.9)
        ax.set_xlabel(f"{dlabel} score", fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(dlabel, fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle("Per-Instance Score Distributions — Stub Baselines\n"
                 "(dashed lines = means; not saturated → room for LLM improvement)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--data",    default=DEFAULT_DATA)
    ap.add_argument("--out",     default=DEFAULT_OUT)
    ap.add_argument("--format",  default="pdf", choices=["pdf", "png", "svg"],
                    help="Output file format (default: pdf)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ext = args.format

    print("Loading data...")
    llm_models = load_results(args.results)

    # Load every stub file that exists
    stub_entries = []
    stub_data_by_name: Dict[str, Dict] = {}
    for name, path in STUB_FILES.items():
        if os.path.exists(path):
            data = load_stub(path)
            stub_data_by_name[name] = data
            stub_entries.append(stub_to_model_entry(data, name))
        else:
            print(f"  WARNING: stub file not found: {path}")

    all_models = llm_models + stub_entries

    print("Loading instances (for Appendix A)...")
    instances = load_instances(args.data)

    print("Generating figures...")

    fig3_heatmap(all_models, os.path.join(args.out, f"fig3_heatmap.{ext}"))
    fig4_scatter(all_models, os.path.join(args.out, f"fig4_scatter.{ext}"))
    fig5_intra_family(llm_models, os.path.join(args.out, f"fig5_intra_family.{ext}"))
    figA_oracle_span(instances, os.path.join(args.out, f"figA_oracle_span.{ext}"))

    # figB needs per-instance data — use whichever stubs have it
    sc_data = stub_data_by_name.get("stub-cautious", {})
    sp_data = stub_data_by_name.get("stub-probe-heavy", {})
    figB_dimension_dists(sc_data, sp_data, os.path.join(args.out, f"figB_dimension_dists.{ext}"))

    print(f"\nAll figures written to: {args.out}/")


if __name__ == "__main__":
    main()
