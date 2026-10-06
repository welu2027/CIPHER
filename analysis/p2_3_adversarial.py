"""2.3 Alternative adversarial worlds for the contingency component (C7).

The published B builds ONE adversarial world (each hidden rule's effect replaced
by zero_flux on its target). Here the saved final and contingency plans are
re-simulated under three alternatives; hidden rules keep their true triggers
and only their effects change, as in zero_flux:

 (a) true      the actual full world (does the contingency beat the final plan
               when the hidden rules are what they really are?)
 (b) random    each hidden effect resampled with the generator's own
               _random_effect (kind, target, delta, source); 20 worlds per
               instance, seeded by (instance seed, sample), identical for all
               models; B averaged over the 20.
 (c) maxdmg    hidden effects chosen from the generator's effect space to MINIMISE
               the final plan's objective. Exhaustive enumeration with 1-2 hidden
               rules (<= 60^2 worlds). With 3 hidden rules, block-coordinate
               descent: each step jointly optimises a PAIR of hidden effects
               exhaustively with the third fixed, cycling over the 3 pairs until no
               improvement (<= 4 passes), from two starts (zero_flux, true effects).
               Its gap to full exhaustive search (60^3 worlds) is measured on a
               random sample of 3-hidden-rule cases.

B under each world uses the scorer's bucket rule (0 / 0.1 / 0.3 / 0.5+gain/span,
span = stored oracle best - worst). Executive variant = 0.5*A + 0.5*B_variant.
Scorer-independent; results are cached in analysis/out/cache/adv_variants.pkl.

Outputs: analysis/out/<scorer>/p2_3/
"""

from __future__ import annotations

import itertools
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

N_RANDOM = 20
MAX_PASSES = 4
VALIDATION_SAMPLE = 40

_RECS = None


def _init():
    global _RECS
    from reanalysis_common import load_records
    _RECS = {r["id"]: r for r in load_records()}


def effect_space(n):
    from cipher.world import Effect
    out = []
    for t in range(n):
        for d in (-2, -1, 1, 2, 3):
            out.append(Effect("flux_add", t, d, -1))
            out.append(Effect("phase_add", t, d, -1))
        for s in range(n):
            if s != t:
                out.append(Effect("align_phase", t, 0, s))
        out.append(Effect("swap_pf", t, 0, -1))
        out.append(Effect("zero_flux", t, 0, -1))
    return out


def _obj(inst, effects, actions):
    from reanalysis_sim import world_with_hidden_effects
    w = world_with_hidden_effects(inst, effects)
    return w.objective(w.execute(actions)), w


def max_damage(inst, actions, exact=False):
    from cipher.world import Effect
    hidden = list(inst.hidden_rule_indices)
    space = effect_space(inst.world.initial.n)
    if exact or len(hidden) <= 2:
        best_v, best_e = None, None
        for combo in itertools.product(space, repeat=len(hidden)):
            e = dict(zip(hidden, combo))
            v, _ = _obj(inst, e, actions)
            if best_v is None or v < best_v:
                best_v, best_e = v, e
        return best_v, best_e
    starts = [
        {i: Effect("zero_flux", inst.world.rules[i].effect.target, 0, -1) for i in hidden},
        {i: inst.world.rules[i].effect for i in hidden},
    ]
    best_v, best_e = None, None
    for cur in starts:
        cur = dict(cur)
        val, _ = _obj(inst, cur, actions)
        for _ in range(MAX_PASSES):
            improved = False
            for i, j in itertools.combinations(hidden, 2):
                for ei, ej in itertools.product(space, repeat=2):
                    cand = {**cur, i: ei, j: ej}
                    v, _ = _obj(inst, cand, actions)
                    if v < val:
                        val, cur, improved = v, cand, True
            if not improved:
                break
        if best_v is None or val < best_v:
            best_v, best_e = val, cur
    return best_v, best_e


def _bucket(alt_obj, fin_obj, span, has_alt, differs):
    from reanalysis_sim import contingency_bucket_score
    return contingency_bucket_score(alt_obj, fin_obj, span, has_alt, differs)


def work(job):
    """job = (model, [(iid, probes, final, alt), ...]) with actions as (kind,i,j) tuples."""
    from cipher.generator import _random_effect
    from cipher.world import Action
    from reanalysis_common import instance_from_record
    from reanalysis_sim import combined
    model, items = job
    out = {}
    md_cache = {}
    for iid, probes, final, alt in items:
        rec = _RECS[iid]
        inst = instance_from_record(rec)
        h = inst.world.horizon
        P = [Action(*a) for a in probes]
        F = [Action(*a) for a in final]
        Al = [Action(*a) for a in alt]
        fin_c = combined(P, F, h)
        alt_c = combined(P, Al, h) if Al else fin_c
        has_alt, differs = bool(Al), [a for a in alt] != [a for a in final]
        span = max(1, rec["hidden"]["oracle_best"] - rec["hidden"]["oracle_worst"])
        r = {}
        # (a) true world
        w = inst.world
        ft, at = w.objective(w.execute(fin_c)), w.objective(w.execute(alt_c))
        r["B_true"] = _bucket(at, ft, span, has_alt, differs)
        r["alt_beats_final_true"] = has_alt and at > ft
        # (b) random effect substitution
        bs, beats = [], []
        for s in range(N_RANDOM):
            rng = random.Random(f"{inst.seed}:{s}")
            eff = {i: _random_effect(rng, inst.world.initial.n) for i in inst.hidden_rule_indices}
            fo, wr = _obj(inst, eff, fin_c)
            ao = wr.objective(wr.execute(alt_c))
            bs.append(_bucket(ao, fo, span, has_alt, differs))
            beats.append(has_alt and ao > fo)
        r["B_random"] = float(np.mean(bs))
        r["alt_beats_final_random"] = float(np.mean(beats))
        # (c) max damage against the final plan
        key = (iid, tuple(fin_c))
        if key not in md_cache:
            md_cache[key] = max_damage(inst, fin_c)
        fo, eff = md_cache[key]
        _, wm = _obj(inst, eff, fin_c)
        ao = wm.objective(wm.execute(alt_c))
        r["B_maxdmg"] = _bucket(ao, fo, span, has_alt, differs)
        r["alt_beats_final_maxdmg"] = has_alt and ao > fo
        r["maxdmg_final_raw"] = fo
        out[iid] = r
    return model, out


def validate(job):
    """Block-coordinate descent vs exhaustive search on 3-hidden-rule cases."""
    from cipher.world import Action
    from reanalysis_common import instance_from_record
    from reanalysis_sim import combined
    gaps = []
    for iid, probes, final, alt in job:
        inst = instance_from_record(_RECS[iid])
        fin_c = combined([Action(*a) for a in probes], [Action(*a) for a in final], inst.world.horizon)
        cd, _ = max_damage(inst, fin_c)
        ex, _ = max_damage(inst, fin_c, exact=True)
        gaps.append(cd - ex)
    return gaps


def _items(per):
    tup = lambda acts: [(a.kind, a.i, a.j) for a in acts]
    return [(iid, tup(v["parsed"].exploratory_actions), tup(v["parsed"].final_plan),
             tup(v["parsed"].self_judgment.alternative_plan))
            for iid, v in per.items() if v["parsed"] is not None]


def main():
    from reanalysis_common import CACHE_DIR, MODELS, SHARED_CACHE, STUBS, ensure_out, write_csv
    from reanalysis_stats import bootstrap_corr_ci, corr, fmt_p
    from scipy import stats

    out = ensure_out("p2_3")
    with open(os.path.join(CACHE_DIR, "scored.pkl"), "rb") as f:
        scored = pickle.load(f)
    with open(os.path.join(CACHE_DIR, "stubs.pkl"), "rb") as f:
        stubs = pickle.load(f)
    with open(os.path.join(CACHE_DIR, "features_p2_1.pkl"), "rb") as f:
        f21 = pickle.load(f)
    sources = {**{n: scored[n] for n in MODELS}, **{n: stubs[n] for n in STUBS}}

    cache = os.path.join(SHARED_CACHE, "adv_variants.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            res, gaps = pickle.load(f)
    else:
        t0 = time.time()
        jobs = []
        for name, per in sources.items():
            items = _items(per)
            for k in range(0, len(items), 100):
                jobs.append((name, items[k:k + 100]))
        res = {n: {} for n in sources}
        with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init) as ex:
            for name, part in ex.map(work, jobs):
                res[name].update(part)
            # validation sample: 3-hidden-rule (hard) cases from all LLMs
            _init()
            med = [it for n in MODELS for it in _items(scored[n]) if _RECS[it[0]]["difficulty"] == "hard"]
            rng = random.Random(42)
            sample = rng.sample(med, min(VALIDATION_SAMPLE, len(med)))
            gaps = [g for part in ex.map(validate, [sample[i::os.cpu_count()] for i in range(os.cpu_count())]) for g in part]
        with open(cache, "wb") as f:
            pickle.dump((res, gaps), f)
        print(f"simulated in {time.time() - t0:.0f}s")

    variants = ["zero_flux", "true", "random", "maxdmg"]
    table = []
    for name in sources:
        fe = f21["llm"][name] if name in MODELS else f21["stub"][name]
        ids = [i for i in res[name] if i in fe]
        m = lambda vals: float(np.mean(vals))
        row = {"model": name, "n": len(ids), "objective": m([fe[i]["objective"] for i in ids]), "A": m([fe[i]["A"] for i in ids])}
        row["B_zero_flux"] = m([fe[i]["B"] for i in ids])
        for v in variants[1:]:
            row[f"B_{v}"] = m([res[name][i][f"B_{v}"] for i in ids])
            row[f"frac_alt_beats_final_{v}"] = m([float(res[name][i][f"alt_beats_final_{v}"]) for i in ids])
        for v in variants:
            row[f"executive_{v}"] = 0.5 * row["A"] + 0.5 * row[f"B_{v}"]
        table.append(row)
    write_csv(os.path.join(out, "B_by_adversary.csv"), table)

    llm = [r for r in table if r["model"] in MODELS]
    agree = []
    for a, b in itertools.combinations(variants, 2):
        s = stats.spearmanr([r[f"B_{a}"] for r in llm], [r[f"B_{b}"] for r in llm])
        k = stats.kendalltau([r[f"B_{a}"] for r in llm], [r[f"B_{b}"] for r in llm])
        agree.append({"a": a, "b": b, "spearman_rho": float(s.statistic), "spearman_p": float(s.pvalue),
                      "kendall_tau": float(k.statistic)})
    write_csv(os.path.join(out, "rank_agreement.csv"), agree)

    crow = []
    for sname, rr in {"n=16": llm, "n=15 excl. Gemini 2.5 Pro": [r for r in llm if r["model"] != "Gemini 2.5 Pro"]}.items():
        for v in variants:
            for y in [f"B_{v}", f"executive_{v}"]:
                xs, ys = [r["objective"] for r in rr], [r[y] for r in rr]
                c = corr(xs, ys)
                c.update(bootstrap_corr_ci(xs, ys))
                crow.append({"set": sname, "x": "objective", "y": y, **c})
    write_csv(os.path.join(out, "correlations.csv"), crow)

    lines = []
    def p(s=""):
        print(s)
        lines.append(s)
    g = np.array(gaps)
    p(f"Max-damage validation (block-coordinate descent vs exhaustive, {len(g)} three-hidden-rule cases): "
      f"exact match {np.mean(g == 0):.1%}, mean gap {g.mean():.3f} objective points, max gap {g.max() if len(g) else 0}")
    p(f"\n{'model':26s} {'obj':>6} {'A':>6} | {'B_zf':>6} {'B_true':>6} {'B_rand':>6} {'B_maxd':>6} | "
      f"{'beat_true':>9} {'beat_rand':>9} {'beat_maxd':>9}")
    for r in table:
        p(f"{r['model']:26s} {r['objective']:6.3f} {r['A']:6.3f} | {r['B_zero_flux']:6.3f} {r['B_true']:6.3f} {r['B_random']:6.3f} "
          f"{r['B_maxdmg']:6.3f} | {r['frac_alt_beats_final_true']:9.2f} {r['frac_alt_beats_final_random']:9.2f} {r['frac_alt_beats_final_maxdmg']:9.2f}")
    p("\nRank agreement of model-level B between adversaries (LLMs, n=16):")
    for a in agree:
        p(f"   {a['a']:>9s} vs {a['b']:<9s} Spearman rho={a['spearman_rho']:+.3f} (p={fmt_p(a['spearman_p'])})  Kendall tau={a['kendall_tau']:+.3f}")
    p("\nObjective ~ B / executive under each adversary:")
    for c in crow:
        p(f"   [{c['set']:26s}] obj ~ {c['y']:<20s} r={c['pearson_r']:+.3f} (p={fmt_p(c['pearson_p'])}) "
          f"boot95[{c['pearson_ci_lo']:+.2f},{c['pearson_ci_hi']:+.2f}]  rho={c['spearman_rho']:+.3f} (p={fmt_p(c['spearman_p'])})")
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    tex = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
           r"Model & Obj & B$_{\text{zero\_flux}}$ & B$_{\text{true}}$ & B$_{\text{random}}$ & B$_{\text{max-dmg}}$ & "
           r"Exec$_{\text{true}}$ & Exec$_{\text{random}}$ & Exec$_{\text{max-dmg}}$ \\", r"\midrule"]
    for r in sorted(llm, key=lambda r: -r["objective"]) + [r for r in table if r["model"] in STUBS]:
        tex.append(f"{r['model']} & {r['objective']:.3f} & {r['B_zero_flux']:.3f} & {r['B_true']:.3f} & {r['B_random']:.3f} & "
                   f"{r['B_maxdmg']:.3f} & {r['executive_true']:.3f} & {r['executive_random']:.3f} & {r['executive_maxdmg']:.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "B_by_adversary.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
