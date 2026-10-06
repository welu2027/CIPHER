"""Part 1 - inventory of saved responses and reproduction of paper Table 5.

For every model:
  * how many of the 1,000 instances have a saved response (subrun) at all,
  * how many saved replies parse, and how many carry schema errors,
  * whether re-scoring each saved reply with cipher/scorer.py reproduces the
    per-instance scores Kaggle stored at run time,
  * whether Table 5 is the mean over saved responses ("valid-only") or the
    mean over 1,000 with missing responses scored zero (the Kaggle aggregate).

Outputs (analysis/out/p1/):
  inventory.csv, table5_reproduction.csv, table5_reproduction.md
  ../cache/scored.pkl  (per-instance parsed responses + rescored dims, used by Part 2)
"""

from __future__ import annotations

import os
import pickle
from collections import Counter

from reanalysis_common import (
    CACHE_DIR, DIMS, MODELS, SCORER, ensure_out, instance_from_record, load_all_models,
    load_records, parse_reply, score_fn, write_csv,
)
from cipher.scorer_v2 import CONFIDENCE_RULES, LabelAudit, calibration_v2

# Paper Table 5 (NeurIPS submission #3009, Appendix A), transcribed from the PDF.
# Used only as the reproduction target.
PAPER_TABLE5 = {
    "Claude Opus 4.6":          (0.684, 0.566, 0.902, 0.672, 0.630),
    "Claude Opus 4.5":          (0.663, 0.503, 0.877, 0.676, 0.664),
    "Claude Sonnet 4.6":        (0.657, 0.563, 0.787, 0.674, 0.644),
    "Gemini 2.5 Pro":           (0.643, 0.669, 0.881, 0.652, 0.290),
    "GPT-5.4 Nano":             (0.632, 0.488, 0.837, 0.644, 0.617),
    "GPT-5.4 mini":             (0.629, 0.485, 0.844, 0.672, 0.567),
    "Claude Opus 4.7":          (0.626, 0.506, 0.763, 0.677, 0.613),
    "Gemini 3 Flash Preview":   (0.624, 0.749, 0.758, 0.676, 0.187),
    "Gemini 3.1 Pro Preview":   (0.622, 0.853, 0.753, 0.553, 0.121),
    "Qwen 3 Next 80B Instruct": (0.618, 0.483, 0.839, 0.622, 0.574),
    "Gemini 2.5 Flash":         (0.617, 0.654, 0.808, 0.625, 0.305),
    "DeepSeek V3.2":            (0.606, 0.478, 0.763, 0.672, 0.567),
    "Claude 4.5 Haiku":         (0.605, 0.500, 0.750, 0.676, 0.534),
    "Gemma 4 31B":              (0.603, 0.737, 0.770, 0.676, 0.087),
    "GPT-5.5":                  (0.586, 0.765, 0.766, 0.505, 0.131),
    "GPT-5.4":                  (0.554, 0.490, 0.846, 0.677, 0.179),
}

TOL = 1e-9


def main() -> None:
    out = ensure_out("p1")
    records = load_records()
    rec_by_id = {r["id"]: r for r in records}
    models = load_all_models()

    inv_rows, repro_rows = [], []
    scored = {}
    for name in MODELS:
        m = models[name]
        saved = m["saved"]
        status = Counter()
        schema_err = 0
        mismatch = Counter()
        audit = LabelAudit()
        per = {}
        for iid, sv in saved.items():
            rec = rec_by_id[iid]
            parsed, raw, st = parse_reply(sv.reply)
            status[st] += 1
            if parsed is None:
                per[iid] = {"status": st, "parsed": None, "raw": raw,
                            "scores": {d: 0.0 for d in DIMS}, "stored": sv.stored}
                continue
            inst = instance_from_record(rec)
            bd = score_fn(parsed, inst, rec["hidden"]["oracle_best"], rec["hidden"]["oracle_worst"],
                          audit=audit if SCORER == "v2" else None)
            if parsed.errors:
                schema_err += 1
            for d in DIMS:
                if abs(bd[d] - float(sv.stored.get(d, -1))) > TOL:
                    mismatch[d] += 1
            per[iid] = {"status": st, "parsed": parsed, "raw": raw, "scores": bd,
                        "stored": sv.stored, "n_schema_errors": len(parsed.errors)}
        scored[name] = per

        n_saved = len(saved)
        n_missing = 1000 - n_saved
        stored_sum = {d: sum(float(v["stored"].get(d, 0.0)) for v in per.values()) for d in DIMS}
        resc_sum = {d: sum(v["scores"][d] for v in per.values()) for d in DIMS}
        inv_rows.append({
            "model": name, "slug": m["slug"], "subruns_in_file": m["n_subruns"],
            "saved_matched": n_saved, "missing_of_1000": n_missing,
            "prompt_mojibake": sum(sv.prompt_mojibake for sv in saved.values()),
            "parse_ok": status["ok"], "parse_fail_among_saved": n_saved - status["ok"],
            "schema_errors_among_saved": schema_err,
            "scorer": SCORER,
            "rescore_mismatch_vs_stored_v1_any_dim": sum(mismatch.values()),
            **({"v2_claims": audit.claims, "v2_claims_mapped": audit.mapped,
                "v2_duplicate_claims_dropped": audit.duplicates, "v2_label_conflicts": audit.conflicts,
                "v2_gt_items_unanswered": audit.unmatched_gt} if SCORER == "v2" else {}),
            "kaggle_aggregate_composite_n1000": m["kaggle_aggregate"],
            "stored_sum_composite_div1000": stored_sum["composite"] / 1000,
        })
        paper = PAPER_TABLE5[name]
        for mode in ("valid_only", "missing_as_zero"):
            denom = n_saved if mode == "valid_only" else 1000
            row = {"model": name, "mode": mode, "n": denom}
            ok = True
            for k, d in enumerate(DIMS):
                v = resc_sum[d] / denom
                row[d] = v
                row[f"paper_{d}"] = paper[k]
                row[f"diff_{d}"] = round(v, 3) - paper[k]
                ok &= abs(round(v, 3) - paper[k]) < 1e-9
            row["reproduces_3dp"] = ok
            repro_rows.append(row)

    with open(os.path.join(CACHE_DIR, "scored.pkl"), "wb") as f:
        pickle.dump(scored, f)

    if SCORER == "v2":  # sensitivity of calibration to the confidence reading
        sem_rows = []
        for name in MODELS:
            row = {"model": name}
            for rule in CONFIDENCE_RULES:
                vals = [calibration_v2(v["parsed"], instance_from_record(rec_by_id[i]), confidence_rule=rule)
                        for i, v in scored[name].items() if v["parsed"] is not None]
                row[f"calibration_{rule}"] = sum(vals) / len(vals)
            sem_rows.append(row)
        write_csv(os.path.join(out, "calibration_confidence_semantics.csv"), sem_rows)
    write_csv(os.path.join(out, "inventory.csv"), inv_rows)
    write_csv(os.path.join(out, "table5_reproduction.csv"), repro_rows)

    lines = [f"# Part 1 - Inventory and Table 5 reproduction (scorer {SCORER}; "
             f"mismatch column = per-instance differences from the v1 scores stored at run time)", "",
             "| model | saved | missing | parse fail (saved) | schema-err (saved) | rescore mismatches | Kaggle agg (n=1000) | stored sum/1000 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in inv_rows:
        lines.append(f"| {r['model']} | {r['saved_matched']} | {r['missing_of_1000']} | {r['parse_fail_among_saved']} | "
                     f"{r['schema_errors_among_saved']} | {r['rescore_mismatch_vs_stored_v1_any_dim']} | "
                     f"{r['kaggle_aggregate_composite_n1000']:.4f} | {r['stored_sum_composite_div1000']:.4f} |")
    lines += ["", "## Table 5 reproduction (rescored from saved replies)", "",
              "| model | mode | n | comp | obj | cal | att | exec | reproduces (3dp) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in repro_rows:
        cells = " | ".join(f"{r[d]:.3f} ({r['diff_'+d]:+.3f})" for d in DIMS)
        lines.append(f"| {r['model']} | {r['mode']} | {r['n']} | {cells} | {r['reproduces_3dp']} |")
    with open(os.path.join(out, "table5_reproduction.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
