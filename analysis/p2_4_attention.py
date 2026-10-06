"""2.4 Attention baselines (C4).

Per instance, three fixed rankings are scored with the paper's attention rule:
  random (expected value over all permutations), given order (H0, H1, ...),
  reverse order.
Tests whether H-label order carries information about ground-truth impact.
H-labels are assigned in ascending rule index, and rules fire in index order,
so H-label order == the hidden rules' order in the firing sequence.

Ground-truth impact is recomputed exactly as cipher/generator.py does (ablate
each hidden rule from the beam-64 oracle plan on the full world); the resulting
ranking is checked against the stored true_unknown_ranking.

Outputs: analysis/out/p2_4/
"""

from __future__ import annotations

import itertools
import os
import pickle
from collections import Counter

import numpy as np

from reanalysis_common import (
    CACHE_DIR, MODELS, SCORER, SHARED_CACHE, STUBS, WEIGHTS, ensure_out, instance_from_record,
    load_records, write_csv,
)
from reanalysis_stats import corr, fmt_p
from cipher.generator import _oracle_plan
from cipher.scorer import _attention as _attention_v1
from cipher.scorer_v2 import attention_v2, normalized_ranking

_attention = _attention_v1 if SCORER == "v1" else attention_v2
from cipher.schema import ParsedResponse, SelfJudgment
from cipher.world import World


def attention_of(ranking, inst):
    resp = ParsedResponse([], list(ranking), [], [], SelfJudgment(50, [], "", []))
    return _attention(resp, inst)


def impacts(rec_by_id):
    cache = os.path.join(SHARED_CACHE, "impacts.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    out = {}
    for iid, rec in rec_by_id.items():
        inst = instance_from_record(rec)
        w = inst.world
        plan = _oracle_plan(w)
        base = w.objective(w.execute(plan))
        imp = []
        for idx in inst.hidden_rule_indices:
            red = World(initial=w.initial, rules=tuple(r for i, r in enumerate(w.rules) if i != idx), horizon=w.horizon)
            imp.append(abs(base - red.objective(red.execute(plan))))
        out[iid] = imp
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    out = ensure_out("p2_4")
    records = load_records()
    rec_by_id = {r["id"]: r for r in records}
    imp = impacts(rec_by_id)

    # ---- instance-level baselines + H-order informativeness ----
    inst_rows = []
    for iid, rec in rec_by_id.items():
        inst = instance_from_record(rec)
        nh = len(inst.hidden_rule_indices)
        labels = [f"H{i}" for i in range(nh)]
        im = imp[iid]
        # verify recomputed ranking == stored (stable sort, ties keep H order)
        order = sorted(range(nh), key=lambda k: -im[k])
        assert [inst.world.rules[inst.hidden_rule_indices[k]].name for k in order] == inst.true_unknown_ranking, iid
        perms = list(itertools.permutations(labels))
        rand = float(np.mean([attention_of(p, inst) for p in perms]))
        given = attention_of(labels, inst)
        rev = attention_of(labels[::-1], inst)
        all_tied = nh >= 2 and len(set(im)) == 1
        any_tie = nh >= 2 and len(set(im)) < nh
        strict_pairs = [(a, b) for a, b in itertools.combinations(range(nh), 2) if im[a] != im[b]]
        conc = sum(im[a] > im[b] for a, b in strict_pairs)  # earlier label has larger impact
        inst_rows.append({
            "id": iid, "difficulty": rec["difficulty"], "n_hidden": nh,
            "att_random_expected": rand, "att_given_order": given, "att_reverse_order": rev,
            "truth_equals_given_order": order == list(range(nh)),
            "impacts_all_tied": all_tied, "impacts_any_tie": any_tie,
            "all_impacts_zero": nh >= 2 and max(im) == 0,
            "strict_pairs": len(strict_pairs), "strict_pairs_earlier_label_higher_impact": conc,
            "hidden_rule_indices": ",".join(map(str, inst.hidden_rule_indices)),
        })
    write_csv(os.path.join(out, "instance_baselines.csv"), inst_rows)

    def m(rows, k):
        return float(np.mean([float(r[k]) for r in rows])) if rows else float("nan")
    mh = [r for r in inst_rows if r["n_hidden"] >= 2]
    base_summary = {
        "all": {k: m(inst_rows, k) for k in ["att_random_expected", "att_given_order", "att_reverse_order"]},
        "medium+hard": {k: m(mh, k) for k in ["att_random_expected", "att_given_order", "att_reverse_order",
                                               "truth_equals_given_order", "impacts_all_tied",
                                               "impacts_any_tie", "all_impacts_zero"]},
    }
    for d in ["easy", "medium", "hard"]:
        rr = [r for r in inst_rows if r["difficulty"] == d]
        base_summary[d] = {k: m(rr, k) for k in ["att_random_expected", "att_given_order", "att_reverse_order",
                                                   "truth_equals_given_order", "impacts_all_tied", "all_impacts_zero"]}
    sp = sum(r["strict_pairs"] for r in mh)
    sc = sum(r["strict_pairs_earlier_label_higher_impact"] for r in mh)
    from scipy import stats
    binom = stats.binomtest(sc, sp, 0.5)
    base_summary["strict_pairs"] = {"n_pairs": sp, "frac_earlier_label_higher_impact": sc / sp if sp else float("nan"),
                                    "binom_p": float(binom.pvalue)}

    # ---- per-model ----
    with open(os.path.join(CACHE_DIR, "scored.pkl"), "rb") as f:
        scored = pickle.load(f)
    with open(os.path.join(CACHE_DIR, "stubs.pkl"), "rb") as f:
        stubs = pickle.load(f)
    by_id = {r["id"]: r for r in inst_rows}
    model_rows = []
    for name, per in [(n, scored[n]) for n in MODELS] + [(n, stubs[n]) for n in STUBS]:
        rows = []
        for iid, v in per.items():
            if v["parsed"] is None:
                continue
            nh = by_id[iid]["n_hidden"]
            labels = [f"H{i}" for i in range(nh)]
            if SCORER == "v1":
                rk = [x.strip() for x in v["parsed"].critical_unknowns_ranked]
            else:  # normalised to hidden-rule names, mapped back to H-labels
                inst_ = instance_from_record(rec_by_id[iid])
                hid_names = [inst_.world.rules[i].name for i in inst_.hidden_rule_indices]
                rk = [f"H{hid_names.index(nm)}" for nm in normalized_ranking(v["parsed"], inst_)]
            sc_ = v["scores"]
            comp_no_att = (WEIGHTS["objective"] * sc_["objective"] + WEIGHTS["calibration"] * sc_["calibration"]
                           + WEIGHTS["executive"] * sc_["executive"]) / (1 - WEIGHTS["attention"])
            rows.append({"nh": nh, "att": sc_["attention"], "given": by_id[iid]["att_given_order"],
                         "identical_given": rk == labels, "identical_reverse": nh >= 2 and rk == labels[::-1],
                         "comp": sc_["composite"], "comp_no_att": comp_no_att})
        mhr = [r for r in rows if r["nh"] >= 2]
        model_rows.append({
            "model": name, "n": len(rows), "n_med_hard": len(mhr),
            "attention_all": m(rows, "att"), "given_order_baseline_all": m(rows, "given"),
            "attention_med_hard": m(mhr, "att"), "given_order_baseline_med_hard": m(mhr, "given"),
            "att_minus_given_med_hard": m(mhr, "att") - m(mhr, "given"),
            "frac_identical_given_order_all": m(rows, "identical_given"),
            "frac_identical_given_order_med_hard": m(mhr, "identical_given"),
            "frac_identical_reverse_med_hard": m(mhr, "identical_reverse"),
            "composite": m(rows, "comp"), "composite_no_attention": m(rows, "comp_no_att"),
        })
    write_csv(os.path.join(out, "per_model_attention.csv"), model_rows)

    llm = [r for r in model_rows if r["model"] in MODELS]
    rank = lambda key: [x["model"] for x in sorted(llm, key=lambda r: -r[key])]
    tau = stats.kendalltau([r["composite"] for r in llm], [r["composite_no_attention"] for r in llm])
    c_att = corr([r["frac_identical_given_order_med_hard"] for r in llm], [r["attention_med_hard"] for r in llm])

    with open(os.path.join(out, "summary.txt"), "w") as f:
        def p(*a):
            print(*a)
            print(*a, file=f)
        p("== Fixed-ranking baselines (attention score) ==")
        for k, v in base_summary.items():
            p(f"  {k:12s} " + "  ".join(f"{kk}={vv:.3f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in v.items()))
        p(f"  binomial test, earlier H-label has higher impact among strictly ordered pairs: p={fmt_p(base_summary['strict_pairs']['binom_p'])}")
        p("\n== Per model ==")
        p(f"{'model':26s} {'att':>6} {'given':>6} {'att_MH':>6} {'givMH':>6} {'diffMH':>7} {'=given':>6} {'=givMH':>6} {'=revMH':>6} {'comp':>6} {'comp-att':>8}")
        for r in model_rows:
            p(f"{r['model']:26s} {r['attention_all']:6.3f} {r['given_order_baseline_all']:6.3f} {r['attention_med_hard']:6.3f} "
              f"{r['given_order_baseline_med_hard']:6.3f} {r['att_minus_given_med_hard']:+7.3f} {r['frac_identical_given_order_all']:6.2f} "
              f"{r['frac_identical_given_order_med_hard']:6.2f} {r['frac_identical_reverse_med_hard']:6.2f} {r['composite']:6.3f} {r['composite_no_attention']:8.3f}")
        p(f"\nKendall tau (LLM ranking: composite vs composite without attention) = {tau.statistic:.3f} (p={fmt_p(tau.pvalue)})")
        p(f"Across LLMs: corr(frac identical to given order, attention on med+hard): r={c_att['pearson_r']:+.3f} (p={fmt_p(c_att['pearson_p'])})")
        p("Ranking with attention:    " + " > ".join(rank("composite")))
        p("Ranking without attention: " + " > ".join(rank("composite_no_attention")))

    tex = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"Model & Att & Att (M+H) & Given-order (M+H) & $\Delta$ & \% = given (M+H) & Comp. w/o Att \\", r"\midrule"]
    for r in model_rows:
        tex.append(f"{r['model']} & {r['attention_all']:.3f} & {r['attention_med_hard']:.3f} & {r['given_order_baseline_med_hard']:.3f} & "
                   f"{r['att_minus_given_med_hard']:+.3f} & {100*r['frac_identical_given_order_med_hard']:.0f}\\% & {r['composite_no_attention']:.3f} \\\\")
    tex += [r"\midrule",
            f"Fixed: given order & {base_summary['all']['att_given_order']:.3f} & {base_summary['medium+hard']['att_given_order']:.3f} & -- & -- & 100\\% & -- \\\\",
            f"Fixed: reverse order & {base_summary['all']['att_reverse_order']:.3f} & {base_summary['medium+hard']['att_reverse_order']:.3f} & -- & -- & 0\\% & -- \\\\",
            f"Fixed: random (expected) & {base_summary['all']['att_random_expected']:.3f} & {base_summary['medium+hard']['att_random_expected']:.3f} & -- & -- & -- & -- \\\\",
            r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "attention_baselines.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
