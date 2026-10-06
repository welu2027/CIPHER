"""Separate-probe-budget ablation: does the objective-executive anti-correlation
survive when probes no longer share the 7-action plan budget?

Inputs
  results/separate_budget/*.jsonl  - written by notebooks/kaggle_separate_budget.ipynb
                                     (one line per instance: model, id, status, raw reply)
  data/instances_separate_budget.jsonl
  Shared-budget responses for the same instances come from the original runs
  (analysis/out/cache/scored.pkl, built by p1_inventory.py).

Everything is re-scored here from raw replies with cipher/separate_budget.py;
the scores the notebook printed are not used.

Reports (analysis/out/p3/ or analysis/out/p3_dryrun/)
  per_model.csv      shared vs separate: objective, executive, A, B, #probes, with
                     instance-paired deltas and bootstrap 95% CIs
  correlation.csv    objective~executive across models in each condition
                     (Pearson/Spearman, bootstrap CI over models), and the change in r
  summary.txt, separate_budget.tex, obj_exec_conditions.{pdf,png}

Usage
  .venv/bin/python analysis/p3_separate_budget.py              # real runs
  .venv/bin/python analysis/p3_separate_budget.py --dry-run    # pipeline test, no new runs:
      feeds the ORIGINAL replies through the separate-budget scorer. This is the
      "same behaviour, free probes" counterfactual, not a substitute for new runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import runpy
import sys

import numpy as np

from reanalysis_common import (
    CACHE_DIR, HERE, MODELS, ROOT, ensure_out, instance_from_record, load_all_models, parse_reply, write_csv,
)
from reanalysis_stats import N_BOOT, SEED, corr, fmt_p, paired_compare

sys.path.insert(0, ROOT)
from cipher.separate_budget import SCORER_VERSION, score_response_separate  # noqa: E402

SUBSET = os.path.join(ROOT, "data", "instances_separate_budget.jsonl")
RUN_DIR = os.path.join(ROOT, "results", "separate_budget")
DRY_DIR = os.path.join(ROOT, "results", "separate_budget_dryrun")
KEYS = ["objective", "executive", "A", "B", "calibration", "attention", "composite", "n_probes"]


def load_subset():
    with open(SUBSET, encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}


def shared_scored():
    path = os.path.join(CACHE_DIR, "scored.pkl")
    if not os.path.exists(path):
        runpy.run_path(os.path.join(HERE, "p1_inventory.py"), run_name="__main__")
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_model(tag: str, slugs: dict) -> str:
    """Map a notebook model tag to a display name via the original runs' slugs."""
    t = tag.lower().replace("-", "/", 0)
    for name, slug in slugs.items():
        s = (slug or "").lower()
        if t == s or t.replace("-", "/") == s.replace("-", "/") or t in s or s.split("@")[0] in t:
            return name
    return tag


def shared_features(v, rec):
    """Shared-budget scores for one saved response, incl. A/B (cipher/scorer.py rules)."""
    from reanalysis_sim import decompose_executive
    f = decompose_executive(v["parsed"], instance_from_record(rec), rec["hidden"]["oracle_best"], rec["hidden"]["oracle_worst"])
    return {"objective": v["scores"]["objective"], "executive": v["scores"]["executive"], "A": f["A"], "B": f["B"],
            "calibration": v["scores"]["calibration"], "attention": v["scores"]["attention"],
            "composite": v["scores"]["composite"], "n_probes": f["n_probes_effective"]}


def separate_features(reply, rec):
    parsed, _, status = parse_reply(reply or "")
    if parsed is None:
        return None, status
    s = score_response_separate(parsed, instance_from_record(rec), rec["hidden"]["oracle_best"], rec["hidden"]["oracle_worst"])
    return {"objective": s.objective, "executive": s.executive, "A": s.A, "B": s.B, "calibration": s.calibration,
            "attention": s.attention, "composite": s.composite, "n_probes": s.n_probes_used}, "ok"


def write_dry_run(subset, scored):
    """Write notebook-format JSONL from the ORIGINAL replies (pipeline test only)."""
    os.makedirs(DRY_DIR, exist_ok=True)
    models = load_all_models()
    for name in MODELS:
        tag = models[name]["slug"]
        path = os.path.join(DRY_DIR, f"separate_budget__{tag.replace('/', '-')}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for iid in subset:
                sv = models[name]["saved"].get(iid)
                f.write(json.dumps({"model": tag, "id": iid, "status": "ok" if sv else "missing_in_original",
                                    "reply": sv.reply if sv else None, "condition": "DRY_RUN_original_replies"},
                                   ensure_ascii=False) + "\n")
    return DRY_DIR


def boot_r_diff(x1, y1, x2, y2, n_boot=N_BOOT, seed=SEED):
    """Bootstrap over models (jointly) for r(separate) - r(shared)."""
    x1, y1, x2, y2 = map(lambda a: np.asarray(a, float), (x1, y1, x2, y2))
    rng = np.random.default_rng(seed)
    n = len(x1)
    r1s, r2s, ds = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if min(np.std(x1[i]), np.std(y1[i]), np.std(x2[i]), np.std(y2[i])) == 0:
            continue
        r1 = np.corrcoef(x1[i], y1[i])[0, 1]
        r2 = np.corrcoef(x2[i], y2[i])[0, 1]
        r1s.append(r1); r2s.append(r2); ds.append(r2 - r1)
    pc = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"shared_ci": pc(r1s), "separate_ci": pc(r2s), "diff_ci": pc(ds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--runs", default=None, help="directory of notebook JSONL files")
    args = ap.parse_args()

    subset = load_subset()
    scored = shared_scored()
    run_dir = write_dry_run(subset, scored) if args.dry_run else (args.runs or RUN_DIR)
    out = ensure_out("p3_dryrun" if args.dry_run else "p3")
    files = sorted(glob.glob(os.path.join(run_dir, "*.jsonl")))
    if not files:
        sys.exit(f"No separate-budget runs found in {run_dir}. Download the notebook's "
                 f"separate_budget__*.jsonl files there, or use --dry-run.")

    slugs = {n: m["slug"] for n, m in load_all_models().items()}
    per_model, status_rows = {}, []
    for path in files:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        name = resolve_model(rows[0]["model"], slugs)
        sep, statuses = {}, {}
        for r in rows:
            if r["id"] not in subset:
                continue
            feats, st = separate_features(r.get("reply"), subset[r["id"]]) if r.get("reply") else (None, r.get("status", "no_reply"))
            statuses[st] = statuses.get(st, 0) + 1
            if feats:
                sep[r["id"]] = feats
        shared = {}
        if name in scored:
            for iid in subset:
                v = scored[name].get(iid)
                if v and v["parsed"] is not None:
                    shared[iid] = shared_features(v, subset[iid])
        per_model[name] = {"sep": sep, "shared": shared}
        status_rows.append({"model": name, "file": os.path.basename(path), "n_lines": len(rows),
                            **{f"status_{k}": v for k, v in sorted(statuses.items())}})
    write_csv(os.path.join(out, "parse_status.csv"), status_rows)

    # ---- per model: paired deltas on instances valid in both conditions ----
    table = []
    for name, d in per_model.items():
        ids = sorted(set(d["sep"]) & set(d["shared"]))
        row = {"model": name, "family": MODELS.get(name, ("", "unmatched"))[1],
               "n_separate_valid": len(d["sep"]), "n_shared_valid": len(d["shared"]), "n_paired": len(ids)}
        for k in KEYS:
            row[f"shared_{k}"] = float(np.mean([d["shared"][i][k] for i in ids])) if ids else float("nan")
            row[f"separate_{k}"] = float(np.mean([d["sep"][i][k] for i in ids])) if ids else float("nan")
        for k in ["objective", "executive", "A", "B", "n_probes"]:
            if len(ids) >= 2:
                pcmp = paired_compare([d["sep"][i][k] for i in ids], [d["shared"][i][k] for i in ids])
                row[f"delta_{k}"] = pcmp["delta"]
                row[f"delta_{k}_ci_lo"], row[f"delta_{k}_ci_hi"] = pcmp["ci_lo"], pcmp["ci_hi"]
        table.append(row)
    write_csv(os.path.join(out, "per_model.csv"), table)

    # ---- across-model correlation in each condition (same models, same instances) ----
    ok = [r for r in table if r["n_paired"] >= 30 and r["family"] != "unmatched"]
    lines = []
    def p(s=""):
        print(s)
        lines.append(s)
    p(f"Separate-budget ablation ({'DRY RUN: original replies re-scored with free probes' if args.dry_run else 'new runs'}) "
      f"| scorer {SCORER_VERSION} | subset {len(subset)} instances | models with >=30 paired instances: {len(ok)}")
    corr_rows = []
    if len(ok) >= 3:
        for cond in ["shared", "separate"]:
            for yk in ["executive", "A", "B"]:
                c = corr([r[f"{cond}_objective"] for r in ok], [r[f"{cond}_{yk}"] for r in ok])
                corr_rows.append({"condition": cond, "x": "objective", "y": yk, **c})
        bd = boot_r_diff([r["shared_objective"] for r in ok], [r["shared_executive"] for r in ok],
                         [r["separate_objective"] for r in ok], [r["separate_executive"] for r in ok])
        corr_rows.append({"condition": "bootstrap_over_models", "x": "objective", "y": "executive",
                          "shared_r_ci_lo": bd["shared_ci"][0], "shared_r_ci_hi": bd["shared_ci"][1],
                          "separate_r_ci_lo": bd["separate_ci"][0], "separate_r_ci_hi": bd["separate_ci"][1],
                          "diff_r_ci_lo": bd["diff_ci"][0], "diff_r_ci_hi": bd["diff_ci"][1]})
        write_csv(os.path.join(out, "correlation.csv"), corr_rows)
        p("\nObjective ~ executive across models")
        for c in corr_rows[:-1]:
            p(f"  {c['condition']:9s} obj~{c['y']:<9s} r={c['pearson_r']:+.3f} (p={fmt_p(c['pearson_p'])})  "
              f"rho={c['spearman_rho']:+.3f} (p={fmt_p(c['spearman_p'])})  n={c['n']}")
        p(f"  bootstrap 95% CI (10k, over models): shared r [{bd['shared_ci'][0]:+.2f},{bd['shared_ci'][1]:+.2f}]  "
          f"separate r [{bd['separate_ci'][0]:+.2f},{bd['separate_ci'][1]:+.2f}]  "
          f"change in r [{bd['diff_ci'][0]:+.2f},{bd['diff_ci'][1]:+.2f}]")
    else:
        p("Fewer than 3 usable models - correlations not computed.")

    p(f"\n{'model':26s} {'n':>4} {'obj sh':>7} {'obj sep':>7} {'Δobj [95% CI]':>22} {'exec sh':>7} {'exec sep':>8} "
      f"{'Δexec [95% CI]':>22} {'#pr sh':>6} {'#pr sep':>7}")
    for r in table:
        if r["n_paired"] < 2:
            p(f"{r['model']:26s} {r['n_paired']:4d}  (not enough paired instances)")
            continue
        p(f"{r['model']:26s} {r['n_paired']:4d} {r['shared_objective']:7.3f} {r['separate_objective']:7.3f} "
          f"{r['delta_objective']:+.3f} [{r['delta_objective_ci_lo']:+.3f},{r['delta_objective_ci_hi']:+.3f}] "
          f"{r['shared_executive']:7.3f} {r['separate_executive']:8.3f} "
          f"{r['delta_executive']:+.3f} [{r['delta_executive_ci_lo']:+.3f},{r['delta_executive_ci_hi']:+.3f}] "
          f"{r['shared_n_probes']:6.2f} {r['separate_n_probes']:7.2f}")
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    tex = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"Model & $n$ & Obj (shared) & Obj (sep.) & Exec (shared) & Exec (sep.) & $\Delta$Exec [95\% CI] \\", r"\midrule"]
    for r in sorted(ok, key=lambda r: -r["shared_objective"]):
        tex.append(f"{r['model']} & {r['n_paired']} & {r['shared_objective']:.3f} & {r['separate_objective']:.3f} & "
                   f"{r['shared_executive']:.3f} & {r['separate_executive']:.3f} & "
                   f"{r['delta_executive']:+.3f} [{r['delta_executive_ci_lo']:+.3f}, {r['delta_executive_ci_hi']:+.3f}] \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "separate_budget.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    if len(ok) >= 3:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for r in ok:
            ax.annotate("", xy=(r["separate_objective"], r["separate_executive"]),
                        xytext=(r["shared_objective"], r["shared_executive"]),
                        arrowprops=dict(arrowstyle="->", color="0.6", lw=0.8))
            ax.scatter(r["shared_objective"], r["shared_executive"], c="tab:blue", s=18)
            ax.scatter(r["separate_objective"], r["separate_executive"], c="tab:red", s=18)
            ax.annotate(r["model"], (r["separate_objective"], r["separate_executive"]), fontsize=5.5,
                        xytext=(2, 2), textcoords="offset points")
        ax.scatter([], [], c="tab:blue", label="shared budget (original)")
        ax.scatter([], [], c="tab:red", label="separate probe budget")
        ax.set_xlabel("Objective")
        ax.set_ylabel("Executive")
        ax.legend(fontsize=7)
        ax.set_title("DRY RUN (original replies)" if args.dry_run else "Separate probe budget", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "obj_exec_conditions.pdf"))
        fig.savefig(os.path.join(out, "obj_exec_conditions.png"), dpi=150)


if __name__ == "__main__":
    main()
