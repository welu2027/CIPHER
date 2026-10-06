"""2.2 Contingency scored against a fixed reference (C1).

The published B scores a contingency relative to the model's OWN final plan in
the zero_flux adversarial world, so a better primary plan raises the bar. Here:

  B_abs        contingency outcome in the adversarial world, min-max normalised by
               the oracle best / worst on THAT world (beam search, width 64 both
               ways). 0 if no contingency is given (as in B).
  final_abs    the same for the final plan (what the model would get if it did
               not switch).
  vs no-op     contingency vs [wait]*7 in the adversarial world.

Plans are executed exactly as in the scorer: probes first, then the plan,
capped at H=7. These quantities do not depend on the scorer version.

Outputs: analysis/out/<scorer>/p2_2/
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from reanalysis_common import CACHE_DIR, MODELS, SHARED_CACHE, STUBS, ensure_out, instance_from_record, load_records, write_csv
from reanalysis_sim import combined, norm, objective_of, zero_flux_world
from reanalysis_stats import bootstrap_corr_ci, corr, fmt_p
from cipher.optimal import oracle_score
from cipher.scorer import _worst_objective
from cipher.world import Action

BEAM = 64


def adv_oracles(records):
    path = os.path.join(SHARED_CACHE, "adv_oracle.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    out = {}
    for rec in records:
        adv = zero_flux_world(instance_from_record(rec))
        best, _ = oracle_score(adv, beam_width=BEAM)
        worst = _worst_objective(adv, beam_width=BEAM)
        noop = adv.objective(adv.execute([Action("wait")] * adv.horizon))
        out[rec["id"]] = (best, worst, noop)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def features(per, rec_by_id, orac, feats_p21):
    rows = []
    for iid, v in per.items():
        r = v["parsed"]
        if r is None:
            continue
        inst = instance_from_record(rec_by_id[iid])
        adv = zero_flux_world(inst)
        best, worst, noop = orac[iid]
        h = inst.world.horizon
        fin = objective_of(adv, combined(r.exploratory_actions, r.final_plan, h))
        has_alt = bool(r.self_judgment.alternative_plan)
        alt = objective_of(adv, combined(r.exploratory_actions, r.self_judgment.alternative_plan, h)) if has_alt else None
        f = feats_p21[iid]
        rows.append({
            "objective": f["objective"], "A": f["A"], "B": f["B"], "executive": f["executive"],
            "has_alt": has_alt,
            "B_abs": norm(alt, best, worst) if has_alt else 0.0,
            "final_abs": norm(fin, best, worst),
            "noop_abs": norm(noop, best, worst),
            "alt_gt_final": has_alt and alt > fin,
            "alt_gt_noop": has_alt and alt > noop,
            "alt_ge_noop": has_alt and alt >= noop,
            "alt_minus_noop_abs": (norm(alt, best, worst) - norm(noop, best, worst)) if has_alt else float("nan"),
        })
    return rows


def main():
    out = ensure_out("p2_2")
    records = load_records()
    rec_by_id = {r["id"]: r for r in records}
    orac = adv_oracles(records)
    with open(os.path.join(CACHE_DIR, "scored.pkl"), "rb") as f:
        scored = pickle.load(f)
    with open(os.path.join(CACHE_DIR, "stubs.pkl"), "rb") as f:
        stubs = pickle.load(f)
    with open(os.path.join(CACHE_DIR, "features_p2_1.pkl"), "rb") as f:
        f21 = pickle.load(f)

    table = []
    for name, per, fe in [(n, scored[n], f21["llm"][n]) for n in MODELS] + [(n, stubs[n], f21["stub"][n]) for n in STUBS]:
        rows = features(per, rec_by_id, orac, fe)
        m = lambda k: float(np.nanmean([float(x[k]) for x in rows]))
        A, B_abs = m("A"), m("B_abs")
        table.append({"model": name, "n": len(rows), "objective": m("objective"), "A": A, "B": m("B"),
                      "executive": m("executive"), "B_abs": B_abs, "executive_abs": 0.5 * A + 0.5 * B_abs,
                      "final_abs": m("final_abs"), "noop_abs": m("noop_abs"), "has_contingency": m("has_alt"),
                      "frac_alt_beats_final": m("alt_gt_final"), "frac_alt_beats_noop": m("alt_gt_noop"),
                      "frac_alt_at_least_noop": m("alt_ge_noop"), "mean_alt_minus_noop_abs": m("alt_minus_noop_abs")})
    write_csv(os.path.join(out, "contingency_fixed_reference.csv"), table)

    llm = [r for r in table if r["model"] in MODELS]
    sets = {"n=16": llm, "n=15 excl. Gemini 2.5 Pro": [r for r in llm if r["model"] != "Gemini 2.5 Pro"]}
    crow = []
    for sname, rr in sets.items():
        for y in ["executive", "B", "B_abs", "executive_abs", "final_abs", "frac_alt_beats_noop"]:
            xs, ys = [r["objective"] for r in rr], [r[y] for r in rr]
            c = corr(xs, ys)
            c.update(bootstrap_corr_ci(xs, ys))
            crow.append({"set": sname, "x": "objective", "y": y, **c})
    write_csv(os.path.join(out, "correlations.csv"), crow)

    lines = []
    def p(s=""):
        print(s)
        lines.append(s)
    p(f"{'model':26s} {'obj':>6} {'B':>6} {'B_abs':>6} {'fin_abs':>7} {'noop':>6} {'exec':>6} {'exec_abs':>8} {'alt>fin':>7} {'alt>noop':>8} {'alt-noop':>8}")
    for r in table:
        p(f"{r['model']:26s} {r['objective']:6.3f} {r['B']:6.3f} {r['B_abs']:6.3f} {r['final_abs']:7.3f} {r['noop_abs']:6.3f} "
          f"{r['executive']:6.3f} {r['executive_abs']:8.3f} {r['frac_alt_beats_final']:7.2f} {r['frac_alt_beats_noop']:8.2f} {r['mean_alt_minus_noop_abs']:+8.3f}")
    p("")
    for c in crow:
        p(f"[{c['set']:26s}] objective ~ {c['y']:<20s} r={c['pearson_r']:+.3f} (p={fmt_p(c['pearson_p'])}) "
          f"boot95[{c['pearson_ci_lo']:+.2f},{c['pearson_ci_hi']:+.2f}]  rho={c['spearman_rho']:+.3f} (p={fmt_p(c['spearman_p'])})")
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    tex = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
           r"Model & Obj & B & B$_{\text{abs}}$ & Final$_{\text{abs}}$ & Exec & Exec$_{\text{abs}}$ & Cont.$>$no-op \\", r"\midrule"]
    for r in sorted(llm, key=lambda r: -r["objective"]) + [r for r in table if r["model"] in STUBS]:
        tex.append(f"{r['model']} & {r['objective']:.3f} & {r['B']:.3f} & {r['B_abs']:.3f} & {r['final_abs']:.3f} & "
                   f"{r['executive']:.3f} & {r['executive_abs']:.3f} & {100*r['frac_alt_beats_noop']:.0f}\\% \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "contingency_fixed_reference.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
