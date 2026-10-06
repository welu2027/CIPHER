"""2.1 Executive decomposition and probe-free counterfactual objective (C1, C2).

Per model (valid saved responses, as in Table 5):
  A (probe quality) and its three sub-parts, B (contingency) bucket distribution,
  mean #probes, fraction of contingencies naming a hidden rule.
Model-level correlations of objective with executive, A, B.
Probe-free counterfactual: the saved final plan run WITHOUT its probes on the
full world (still capped at H=7, normalised by the stored oracle best/worst).
Note: observe/wait actions are not free - all rules fire after every action -
so removing probes changes when the final plan's actions happen, not just how
many fit.

Outputs: analysis/out/p2_1/
"""

from __future__ import annotations

import os
import pickle
from collections import Counter

import numpy as np

from reanalysis_common import (
    CACHE_DIR, MODELS, SCORER, STUBS, ensure_out, instance_from_record, load_records,
    load_stub, score_fn, write_csv,
)
from reanalysis_sim import decompose_executive, parse_stub, stub_responses
from reanalysis_stats import bootstrap_corr_ci, corr, fmt_p

EXCLUDE_SENSITIVITY = "Gemini 2.5 Pro"  # mis-encoded prompts, 451/1000 missing


def load_scored():
    with open(os.path.join(CACHE_DIR, "scored.pkl"), "rb") as f:
        return pickle.load(f)


def stub_scored(records):
    """Regenerate deterministic stub responses (cached) and score them."""
    cache = os.path.join(CACHE_DIR, "stubs.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    import evaluate  # scripts/evaluate.py (path set by stub_responses)
    agents = stub_responses()
    # v1 reproduces the published baselines (internal rule names); v2 uses prompt labels only.
    evaluate.STUB_LABELS = "internal" if SCORER == "v1" else "prompt"
    out = {}
    for name, agent in agents.items():
        per = {}
        for rec in records:
            inst = instance_from_record(rec)
            raw = agent(inst)
            parsed = parse_stub(raw)
            bd = score_fn(parsed, inst, rec["hidden"]["oracle_best"], rec["hidden"]["oracle_worst"])
            per[rec["id"]] = {"status": "ok", "parsed": parsed, "raw": raw, "scores": bd}
        if SCORER == "v1":  # sanity: v1 must equal the published baseline files
            saved = load_stub(name)["summary"]
            for d in ["composite", "objective", "calibration", "attention", "executive"]:
                m = np.mean([v["scores"][d] for v in per.values()])
                assert abs(m - saved[f"mean_{d}"]) < 1e-9, (name, d, m, saved[f"mean_{d}"])
        out[name] = per
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


def features_for(per, rec_by_id):
    rows = {}
    for iid, v in per.items():
        if v["parsed"] is None:
            continue
        rec = rec_by_id[iid]
        inst = instance_from_record(rec)
        f = decompose_executive(v["parsed"], inst, rec["hidden"]["oracle_best"], rec["hidden"]["oracle_worst"])
        assert abs(f["executive"] - v["scores"]["executive"]) < 1e-9, (iid, f["executive"], v["scores"]["executive"])
        assert abs(f["objective"] - v["scores"]["objective"]) < 1e-9
        f.update({k: v["scores"][k] for k in ["calibration", "attention", "composite"]})
        f["difficulty"] = rec["difficulty"]
        rows[iid] = f
    return rows


def summarise(name, family, feats):
    F = list(feats.values())
    n = len(F)
    mean = lambda k: float(np.mean([float(x[k]) for x in F]))
    bc = Counter(x["B_bucket"] for x in F)
    obj, pf = mean("objective"), mean("objective_probe_free")
    return {
        "model": name, "family": family, "n": n,
        "objective": obj, "calibration": mean("calibration"), "attention": mean("attention"),
        "executive": mean("executive"), "composite": mean("composite"),
        "A": mean("A"), "B": mean("B"),
        "A_has_probe": mean("A_has_probe"), "A_hidden_entity": mean("A_hidden_entity"),
        "A_budget_le3": mean("A_budget_le3"),
        "B_0": bc["0"] / n, "B_0.1": bc["0.1"] / n, "B_0.3": bc["0.3"] / n, "B_0.5-1.0": bc["0.5-1.0"] / n,
        "has_contingency": mean("has_contingency"),
        "contingency_names_hidden": mean("contingency_names_hidden"),
        "mean_probes": mean("n_probes"), "mean_probes_effective": mean("n_probes_effective"),
        "final_truncated": mean("final_truncated"),
        "objective_probe_free": pf, "probe_cost": obj - pf,
    }


def corr_block(rows, xkey, ykeys, label):
    out = []
    xs = [r[xkey] for r in rows]
    for yk in ykeys:
        ys = [r[yk] for r in rows]
        c = corr(xs, ys)
        c.update(bootstrap_corr_ci(xs, ys))
        out.append({"set": label, "x": xkey, "y": yk, **c})
    return out


def main():
    out = ensure_out("p2_1")
    records = load_records()
    rec_by_id = {r["id"]: r for r in records}
    scored = load_scored()

    feats = {name: features_for(scored[name], rec_by_id) for name in MODELS}
    stubs = stub_scored(records)
    stub_feats = {name: features_for(per, rec_by_id) for name, per in stubs.items()}
    with open(os.path.join(CACHE_DIR, "features_p2_1.pkl"), "wb") as f:
        pickle.dump({"llm": feats, "stub": stub_feats}, f)

    llm_rows = [summarise(n, MODELS[n][1], feats[n]) for n in MODELS]
    stub_rows = [summarise(n, "stub", stub_feats[n]) for n in STUBS]
    write_csv(os.path.join(out, "executive_decomposition.csv"), llm_rows + stub_rows)

    ykeys = ["executive", "A", "B"]
    sets = {"LLMs (n=16)": llm_rows,
            f"LLMs excl. {EXCLUDE_SENSITIVITY} (n=15)": [r for r in llm_rows if r["model"] != EXCLUDE_SENSITIVITY]}
    corr_rows = []
    for label, rows in sets.items():
        corr_rows += corr_block(rows, "objective", ykeys, label)
        corr_rows += corr_block(rows, "objective_probe_free", ykeys, label)
        corr_rows += corr_block(rows, "mean_probes", ["objective", "objective_probe_free", "A", "B"], label)
    write_csv(os.path.join(out, "correlations.csv"), corr_rows)

    # Instance-level: does spending budget on probes lower the objective within a model?
    inst_rows = []
    from scipy import stats
    for n in MODELS:
        F = list(feats[n].values())
        npb = [x["n_probes_effective"] for x in F]
        ob = [x["objective"] for x in F]
        rho = stats.spearmanr(npb, ob).statistic if np.std(npb) > 0 else float("nan")
        inst_rows.append({"model": n, "spearman_nprobes_vs_objective_within_model": float(rho),
                          "frac_with_probes": float(np.mean([x > 0 for x in npb]))})
    write_csv(os.path.join(out, "within_model_probe_vs_objective.csv"), inst_rows)

    # LaTeX table
    tex = [r"\begin{tabular}{lrrrrrrrrrrr}", r"\toprule",
           r"Model & Obj & Obj$_{\text{no-probe}}$ & Exec & A & B & has-probe & hid-ent & $\le$3 & \#probes & names-H & B$\geq$0.5 \\",
           r"\midrule"]
    for r in sorted(llm_rows, key=lambda r: -r["executive"]) + stub_rows:
        tex.append(f"{r['model']} & {r['objective']:.3f} & {r['objective_probe_free']:.3f} & {r['executive']:.3f} & "
                   f"{r['A']:.3f} & {r['B']:.3f} & {r['A_has_probe']:.2f} & {r['A_hidden_entity']:.2f} & "
                   f"{r['A_budget_le3']:.2f} & {r['mean_probes']:.2f} & {r['contingency_names_hidden']:.2f} & {r['B_0.5-1.0']:.2f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "executive_decomposition.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (x, y) in zip(axes, [("objective", "executive"), ("objective", "A"), ("objective", "B"),
                                 ("objective_probe_free", "executive")]):
        for r in llm_rows:
            ax.scatter(r[x], r[y], s=22)
            ax.annotate(r["model"].replace("Claude ", "").replace("Gemini ", "Gem ").replace(" Preview", ""),
                        (r[x], r[y]), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
        c = corr([r[x] for r in llm_rows], [r[y] for r in llm_rows])
        ax.set_title(f"{y} vs {x}\nr={c['pearson_r']:+.2f} (p={fmt_p(c['pearson_p'])}), "
                     f"ρ={c['spearman_rho']:+.2f} (p={fmt_p(c['spearman_p'])})", fontsize=8)
        ax.set_xlabel(x, fontsize=8)
        ax.set_ylabel(y, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "executive_decomposition.pdf"))
    fig.savefig(os.path.join(out, "executive_decomposition.png"), dpi=150)

    # Console summary
    print(f"{'model':26s} {'n':>4} {'obj':>6} {'objNP':>6} {'cost':>7} {'exec':>6} {'A':>6} {'B':>6} "
          f"{'has':>5} {'hid':>5} {'<=3':>5} {'#pr':>5} {'B0':>5} {'B.1':>5} {'B.3':>5} {'B.5+':>5} {'namesH':>6} {'trunc':>5}")
    for r in llm_rows + stub_rows:
        print(f"{r['model']:26s} {r['n']:4d} {r['objective']:6.3f} {r['objective_probe_free']:6.3f} {r['probe_cost']:+7.3f} "
              f"{r['executive']:6.3f} {r['A']:6.3f} {r['B']:6.3f} {r['A_has_probe']:5.2f} {r['A_hidden_entity']:5.2f} "
              f"{r['A_budget_le3']:5.2f} {r['mean_probes']:5.2f} {r['B_0']:5.2f} {r['B_0.1']:5.2f} {r['B_0.3']:5.2f} "
              f"{r['B_0.5-1.0']:5.2f} {r['contingency_names_hidden']:6.2f} {r['final_truncated']:5.2f}")
    print()
    for c in corr_rows:
        print(f"{c['set']:38s} {c['x']:>21s} ~ {c['y']:<21s} r={c['pearson_r']:+.3f} p={fmt_p(c['pearson_p']):>8s} "
              f"[{c['pearson_ci_lo']:+.2f},{c['pearson_ci_hi']:+.2f}]  rho={c['spearman_rho']:+.3f} p={fmt_p(c['spearman_p']):>8s}")
    print()
    for r in inst_rows:
        print(f"{r['model']:26s} within-model rho(#probes, obj)={r['spearman_nprobes_vs_objective_within_model']:+.3f}  frac_with_probes={r['frac_with_probes']:.2f}")


if __name__ == "__main__":
    main()
