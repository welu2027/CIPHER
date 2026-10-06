"""2.8 Statistics: recompute every correlation / p-value / paired comparison in the paper.

Model-level table = Table 5 convention (mean over saved, parsed responses),
reproduced exactly in Part 1. Sensitivity variants:
  * missing responses scored 0 over all 1,000 instances (Kaggle run-time aggregate)
  * common instances only (instances every model answered)
  * excluding Gemini 2.5 Pro (mis-encoded prompts; 451/1000 missing)

Paired comparisons (Table 4/7) are paired BY INSTANCE ID. The original
analysis/offline_analysis.py zipped subrun lists positionally; Kaggle stores
subruns out of dataset order, so those pairs were mismatched. The positional
version is recomputed here as well to show it reproduces the published numbers.

Outputs: analysis/out/p2_8/
"""

from __future__ import annotations

import itertools
import os
import pickle

import numpy as np
from scipy import stats

from reanalysis_common import CACHE_DIR, DIMS, MODELS, ensure_out, load_all_models, load_records, write_csv
from reanalysis_stats import bootstrap_corr_ci, corr, fmt_p, paired_compare, pearson_fisher_ci

# Public benchmark values exactly as used for the submitted Appendix B
# (analysis/offline_analysis.py, PUBLIC_BENCHMARKS). NOT re-verified here; see
# CHANGES.md - GPQA equals MMLU-Pro for 4 models, which looks like a transcription error.
PUBLIC = {
    "GPT-5.4 Nano": (0.7717, 0.7717, 0.938), "GPT-5.4 mini": (0.8455, 0.8455, 0.942),
    "GPT-5.4": (0.8748, 0.9167, 0.948), "GPT-5.5": (0.8814, 0.9318, 0.960),
    "Gemini 3 Flash Preview": (0.8859, None, 0.918), "Gemini 3.1 Pro Preview": (0.9099, 0.9545, 0.964),
    "Gemini 2.5 Pro": (0.8406, None, 0.952), "Gemini 2.5 Flash": (0.8366, None, 0.918),
    "Claude Sonnet 4.6": (0.8734, None, 0.938), "Claude Opus 4.6": (0.8911, 0.8965, 0.952),
    "Claude Opus 4.5": (0.8726, 0.8559, 0.930), "Claude Opus 4.7": (0.8987, 0.9015, 0.954),
    "Claude 4.5 Haiku": (0.7872, 0.7872, 0.642), "DeepSeek V3.2": (0.8492, 0.8492, 0.922),
    "Qwen 3 Next 80B Instruct": (None, None, 0.946), "Gemma 4 31B": (None, None, 0.852),
}
BENCH = ["MMLU-Pro", "GPQA", "MATH-500"]

WEIGHT_SCHEMES = {
    "default (35/25/20/20)": (0.35, 0.25, 0.20, 0.20),
    "equal (25/25/25/25)": (0.25, 0.25, 0.25, 0.25),
    "plan-heavy (50/20/15/15)": (0.50, 0.20, 0.15, 0.15),
    "meta-heavy (20/35/25/20)": (0.20, 0.35, 0.25, 0.20),
    "no attention (renorm.)": (0.35 / .8, 0.25 / .8, 0.0, 0.20 / .8),
    "no calibration (renorm.)": (0.35 / .75, 0.0, 0.20 / .75, 0.20 / .75),
}

PAPER = {  # statements in the submitted PDF, for the comparison column
    ("objective", "executive"): "r=-0.81, rho=-0.54, p=0.002 (Sec 6.4); r=-0.80 (Fig 3 caption)",
    ("attention", "executive"): "r=+0.47, p=0.076",
    ("calibration", "executive"): "r=+0.35, p=0.19",
    ("objective", "calibration"): "r=-0.38, p=0.15",
}
PAIRS = [("GPT-5.4 mini", "GPT-5.4"), ("Claude Sonnet 4.6", "Claude Opus 4.7")]


def main():
    out = ensure_out("p2_8")
    records = load_records()
    rec_by_id = {r["id"]: r for r in records}
    with open(os.path.join(CACHE_DIR, "scored.pkl"), "rb") as f:
        scored = pickle.load(f)
    models_raw = load_all_models()
    fam = {n: MODELS[n][1] for n in MODELS}

    per = {n: {iid: v["scores"] for iid, v in scored[n].items() if v["parsed"] is not None} for n in MODELS}
    common = set.intersection(*[set(p) for p in per.values()])

    def table(variant):
        rows = []
        for n in MODELS:
            if variant == "valid_only":
                vals = list(per[n].values()); denom = len(vals)
            elif variant == "missing_as_zero":
                vals = list(per[n].values()); denom = 1000
            elif variant == "common_instances":
                vals = [per[n][i] for i in common]; denom = len(vals)
            rows.append({"model": n, "family": fam[n], "n": denom,
                         **{d: sum(v[d] for v in vals) / denom for d in DIMS}})
        return rows

    variants = {v: table(v) for v in ["valid_only", "missing_as_zero", "common_instances"]}
    write_csv(os.path.join(out, "model_level_table.csv"),
              [{"variant": v, **r} for v, rows in variants.items() for r in rows])

    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p(f"Common instances answered by all 16 models: {len(common)}")

    # ---- inter-dimension correlations ----
    corr_rows = []
    for vname, rows in variants.items():
        sets = {"n=16": rows, "n=15 excl. Gemini 2.5 Pro": [r for r in rows if r["model"] != "Gemini 2.5 Pro"]}
        for sname, rr in sets.items():
            for a, b in itertools.combinations(["objective", "calibration", "attention", "executive"], 2):
                c = corr([r[a] for r in rr], [r[b] for r in rr])
                lo, hi = pearson_fisher_ci(c["pearson_r"], c["n"])
                c.update(bootstrap_corr_ci([r[a] for r in rr], [r[b] for r in rr]))
                corr_rows.append({"variant": vname, "set": sname, "x": a, "y": b, **c,
                                  "pearson_fisher_lo": lo, "pearson_fisher_hi": hi,
                                  "paper": PAPER.get((a, b), "")})
    write_csv(os.path.join(out, "inter_dimension_correlations.csv"), corr_rows)
    p("\n== Inter-dimension correlations over models ==")
    for c in corr_rows:
        if c["set"] != "n=16":
            continue
        p(f"[{c['variant']:16s}] {c['x']:>11s}~{c['y']:<11s} r={c['pearson_r']:+.3f} p={fmt_p(c['pearson_p']):>7s} "
          f"boot95[{c['pearson_ci_lo']:+.2f},{c['pearson_ci_hi']:+.2f}]  rho={c['spearman_rho']:+.3f} p={fmt_p(c['spearman_p']):>7s}   paper: {c['paper']}")
    for c in corr_rows:
        if c["set"] != "n=16" and (c["x"], c["y"]) == ("objective", "executive"):
            p(f"[{c['variant']:16s}] excl. Gemini 2.5 Pro: obj~exec r={c['pearson_r']:+.3f} p={fmt_p(c['pearson_p'])} rho={c['spearman_rho']:+.3f} p={fmt_p(c['spearman_p'])}")

    # ---- obj~exec: leave-one-model-out / leave-one-family-out ----
    rows = variants["valid_only"]
    lo_rows = []
    for n in MODELS:
        rr = [r for r in rows if r["model"] != n]
        c = corr([r["objective"] for r in rr], [r["executive"] for r in rr])
        lo_rows.append({"type": "leave-one-model-out", "left_out": n, **c})
    for f_ in sorted(set(fam.values())):
        rr = [r for r in rows if r["family"] != f_]
        c = corr([r["objective"] for r in rr], [r["executive"] for r in rr])
        lo_rows.append({"type": "leave-one-family-out", "left_out": f_, **c})
    write_csv(os.path.join(out, "obj_exec_leave_out.csv"), lo_rows)
    p("\n== Objective~Executive leave-out (valid_only) ==")
    for t in ["leave-one-model-out", "leave-one-family-out"]:
        rs = [r for r in lo_rows if r["type"] == t]
        p(f"{t}: Pearson r range [{min(r['pearson_r'] for r in rs):+.3f}, {max(r['pearson_r'] for r in rs):+.3f}], "
          f"max p={fmt_p(max(r['pearson_p'] for r in rs))}; Spearman rho range [{min(r['spearman_rho'] for r in rs):+.3f}, "
          f"{max(r['spearman_rho'] for r in rs):+.3f}], max p={fmt_p(max(r['spearman_p'] for r in rs))}")
        if t == "leave-one-family-out":
            for r in rs:
                p(f"   drop {r['left_out']:9s} n={r['n']:2d} r={r['pearson_r']:+.3f} (p={fmt_p(r['pearson_p'])}) rho={r['spearman_rho']:+.3f} (p={fmt_p(r['spearman_p'])})")

    # ---- Table 4: paired comparisons, by instance; and positional (as published) ----
    pc_rows = []
    p("\n== Table 4 recomputed (paired by instance id; bootstrap 10k, seed 42; sign test drops ties) ==")
    for a, b in PAIRS:
        ids = sorted(set(per[a]) & set(per[b]))
        # positional pairing as in offline_analysis.py (file order, zip truncates)
        fa = [models_raw[a]["saved"][k].stored for k in models_raw[a]["saved"]]
        fb = [models_raw[b]["saved"][k].stored for k in models_raw[b]["saved"]]
        n_pos = min(len(fa), len(fb))
        mismatched = sum(1 for x, y in zip(list(models_raw[a]["saved"]), list(models_raw[b]["saved"])) if x != y)
        p(f"{a} vs {b}: instance-paired n={len(ids)}; positional zip n={n_pos} with {mismatched} mismatched instance pairs")
        for d in DIMS:
            r_inst = paired_compare([per[a][i][d] for i in ids], [per[b][i][d] for i in ids])
            r_pos = paired_compare([float(x[d]) for x in fa[:n_pos]], [float(y[d]) for y in fb[:n_pos]])
            pc_rows.append({"comparison": f"{a} vs {b}", "dimension": d, "pairing": "instance", **r_inst})
            pc_rows.append({"comparison": f"{a} vs {b}", "dimension": d, "pairing": "positional (as published)", **r_pos})
            p(f"   {d:11s} INSTANCE Δ={r_inst['delta']:+.3f} [{r_inst['ci_lo']:+.3f},{r_inst['ci_hi']:+.3f}] "
              f"win={100*r_inst['win_frac']:3.0f}% tie={100*r_inst['tie_frac']:3.0f}% loss={100*r_inst['loss_frac']:3.0f}% "
              f"dz={r_inst['d_z']:+.2f} sign-p={fmt_p(r_inst['sign_p_ties_dropped'])}"
              f"  | POSITIONAL Δ={r_pos['delta']:+.3f} [{r_pos['ci_lo']:+.3f},{r_pos['ci_hi']:+.3f}] win={100*r_pos['win_frac']:3.0f}% dz={r_pos['d_z']:+.2f}")
        # per stratum (Table 7) with correct difficulty labels
        for st in ["easy", "medium", "hard"]:
            sid = [i for i in ids if rec_by_id[i]["difficulty"] == st]
            rc = paired_compare([per[a][i]["composite"] for i in sid], [per[b][i]["composite"] for i in sid])
            rcal = paired_compare([per[a][i]["calibration"] for i in sid], [per[b][i]["calibration"] for i in sid])
            rex = paired_compare([per[a][i]["executive"] for i in sid], [per[b][i]["executive"] for i in sid])
            pc_rows.append({"comparison": f"{a} vs {b}", "dimension": f"stratum:{st}", "pairing": "instance",
                            "n_pairs": len(sid), "delta": rc["delta"], "d_z": rc["d_z"],
                            "delta_cal": rcal["delta"], "delta_exec": rex["delta"], "d_z_exec": rex["d_z"]})
            p(f"   [{st:6s} n={len(sid):3d}] ΔComp={rc['delta']:+.3f} dz={rc['d_z']:+.2f}  ΔCal={rcal['delta']:+.3f}  ΔExec={rex['delta']:+.3f} dz={rex['d_z']:+.2f}")
    write_csv(os.path.join(out, "paired_comparisons.csv"), pc_rows)

    # ---- By-difficulty tables (Tables 8 and 11) with correct difficulty labels ----
    bd_rows = []
    for n in MODELS:
        row = {"model": n}
        for st in ["easy", "medium", "hard"]:
            vals = [v for i, v in per[n].items() if rec_by_id[i]["difficulty"] == st]
            row[f"n_{st}"] = len(vals)
            for d in ["composite", "objective"]:
                row[f"{d}_{st}"] = float(np.mean([v[d] for v in vals]))
        row["objective_easy_minus_hard"] = row["objective_easy"] - row["objective_hard"]
        bd_rows.append(row)
    write_csv(os.path.join(out, "by_difficulty.csv"), bd_rows)
    p("\n== By difficulty (correct labels): composite E/M/H, objective E-H delta ==")
    for r in bd_rows:
        p(f"   {r['model']:26s} comp {r['composite_easy']:.3f}/{r['composite_medium']:.3f}/{r['composite_hard']:.3f}  "
          f"obj Δ(E-H)={r['objective_easy_minus_hard']:+.3f}  n={r['n_easy']}/{r['n_medium']}/{r['n_hard']}")

    # ---- Appendix B construct validity ----
    cv_rows = []
    p("\n== Appendix B: construct validity (valid_only model means; * = p<0.05, marked on r and rho independently) ==")
    for d in DIMS:
        cells = []
        for k, bname in enumerate(BENCH):
            names = [n for n in MODELS if PUBLIC[n][k] is not None]
            xs = [PUBLIC[n][k] for n in names]
            ys = [next(r[d] for r in rows if r["model"] == n) for n in names]
            c = corr(xs, ys)
            lo, hi = pearson_fisher_ci(c["pearson_r"], c["n"])
            c.update(bootstrap_corr_ci(xs, ys))
            cv_rows.append({"dimension": d, "benchmark": bname, **c, "pearson_fisher_lo": lo, "pearson_fisher_hi": hi})
            star = lambda pv: "*" if pv < 0.05 else ""
            cells.append(f"{bname}(n={c['n']}): r={c['pearson_r']:+.2f}{star(c['pearson_p'])} [{lo:+.2f},{hi:+.2f}] p={fmt_p(c['pearson_p'])}; "
                         f"rho={c['spearman_rho']:+.2f}{star(c['spearman_p'])} p={fmt_p(c['spearman_p'])}")
        p(f"   {d:11s} " + " | ".join(cells))
    write_csv(os.path.join(out, "construct_validity.csv"), cv_rows)
    dup = [n for n in MODELS if PUBLIC[n][0] is not None and PUBLIC[n][0] == PUBLIC[n][1]]
    p(f"   WARNING: GPQA == MMLU-Pro exactly for {len(dup)} models: {', '.join(dup)}")

    # ---- Weighting robustness (Sec 6.6 / 2.6) ----
    p("\n== Weighting robustness (valid_only; Kendall tau and Spearman vs default ranking) ==")
    base = [sum(w * r[d] for w, d in zip(WEIGHT_SCHEMES["default (35/25/20/20)"], ["objective", "calibration", "attention", "executive"])) for r in rows]
    wr = []
    for sname, w in WEIGHT_SCHEMES.items():
        sc = [sum(wi * r[d] for wi, d in zip(w, ["objective", "calibration", "attention", "executive"])) for r in rows]
        kt = stats.kendalltau(base, sc)
        sr = stats.spearmanr(base, sc)
        top = rows[int(np.argmax(sc))]["model"]
        wr.append({"scheme": sname, "kendall_tau": float(kt.statistic), "kendall_p": float(kt.pvalue),
                   "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue), "top_model": top})
        p(f"   {sname:26s} tau={kt.statistic:+.3f} (p={fmt_p(kt.pvalue)})  rho={sr.statistic:+.3f} (p={fmt_p(sr.pvalue)})  top={top}")
    write_csv(os.path.join(out, "weighting_robustness.csv"), wr)

    # ---- Sec 5.1 floor statements ----
    p("\n== Sec 5.1 / 6.4: distance of objective from the no-op floor (0.480) ==")
    for r in sorted(rows, key=lambda r: r["objective"]):
        p(f"   {r['model']:26s} obj={r['objective']:.3f}  Δfloor={r['objective']-0.480:+.3f}  {'within 0.04' if abs(r['objective']-0.480) <= 0.04 else 'NOT within 0.04'}")

    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
