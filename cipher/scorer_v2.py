"""Scorer v2: label-normalised calibration and attention.

v1 (cipher/scorer.py) matched a model's claims to ground truth by exact string
equality with the generator's internal rule names (R0, R1, ...). The prompt,
however, labels hidden laws H0, H1, ... and names visible rules "[R<idx>] <word>
<greek letter>". Under v1:
  * claims about hidden laws (H-labels) never matched, for any model, and were
    scored as missing (Brier 0.25);
  * claims about visible rules matched only when the model wrote "R0" rather
    than "α", "Edict α", "R0/α", ...
so v1 calibration measured label formatting, not self-knowledge.

v2 maps every label the prompt actually shows to the internal rule name:
  * an explicit visible R-name ("R0", "[R0] Axiom α", "R0/α")  -> that rule
  * an H-label ("H0", "[H0]", "H0_effect_kind")                -> k-th hidden rule
  * a Greek letter ("α", "Canon α", "alpha")                   -> k-th visible rule
The R-names of hidden rules never appear in the prompt, so an "R<idx>" naming a
hidden rule is not mapped. Text that was UTF-8 decoded as cp1252 ("Î±") is
repaired first. If one claim carries both an R-name and a Greek letter that
disagree, the R-name wins (counted in `label_conflicts`). After normalisation,
only the FIRST claim per (rule, component) is scored, so duplicates (e.g. a
model writing both "R0" and "α") cannot count twice.

Everything else is unchanged from v1: Brier-based calibration with a 0.25
penalty per missing (rule, component), the attention concordance rule
(including its partial-credit and tie conventions), objective, and executive.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from .generator import Instance
from .optimal import oracle_score
from .schema import ParsedResponse
from .scorer import _executive, _worst_objective
from .simulator import run_plan

SCORER_VERSION = "v2-label-normalised"
GREEK = "αβγδεζηθικλμν"
GREEK_NAMES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
               "iota", "kappa", "lambda", "mu", "nu"]

_R = re.compile(r"(?<![A-Za-z])R\s*(\d+)(?!\d)")
_H = re.compile(r"(?<![A-Za-z])H\s*(\d+)(?!\d)", re.IGNORECASE)
_GREEK_WORD = re.compile(r"\b(" + "|".join(GREEK_NAMES) + r")\b", re.IGNORECASE)


def _repair(s: str) -> str:
    if any(ch in s for ch in "Îâ"):
        try:
            return s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def label_maps(inst: Instance) -> Tuple[Dict[str, str], Dict[int, str], Dict[str, str]]:
    """(visible R-name -> name, H-index -> hidden name, greek letter -> visible name)."""
    vis = {inst.world.rules[i].name: inst.world.rules[i].name for i in inst.visible_rule_indices}
    hid = {k: inst.world.rules[i].name for k, i in enumerate(inst.hidden_rule_indices)}
    greek = {GREEK[pos]: inst.world.rules[i].name for pos, i in enumerate(inst.visible_rule_indices)}
    return vis, hid, greek


def normalize_label(label: str, inst: Instance, maps=None) -> Tuple[Optional[str], bool]:
    """Return (internal rule name or None, conflict_flag)."""
    vis, hid, greek = maps or label_maps(inst)
    s = _repair(str(label))
    r_name = None
    for m in _R.finditer(s):
        cand = f"R{int(m.group(1))}"
        if cand in vis:
            r_name = cand
            break
    g_name = None
    for ch in s:
        if ch in greek:
            g_name = greek[ch]
            break
    if g_name is None:
        m = _GREEK_WORD.search(s)
        if m:
            g_name = greek.get(GREEK[GREEK_NAMES.index(m.group(1).lower())])
    if r_name:
        return r_name, bool(g_name and g_name != r_name)
    if g_name:
        return g_name, False
    m = _H.search(s)
    if m and int(m.group(1)) in hid:
        return hid[int(m.group(1))], False
    return None, False


@dataclass
class LabelAudit:
    claims: int = 0
    mapped: int = 0
    duplicates: int = 0
    conflicts: int = 0
    unmatched_gt: int = 0


def calibration_v2(resp: ParsedResponse, inst: Instance, audit: Optional[LabelAudit] = None) -> float:
    gt = {(g["rule_name"], g["component"]): g["true_known"] for g in inst.metacog_ground_truth}
    if not gt:
        return 1.0
    maps = label_maps(inst)
    seen = set()
    sq: List[float] = []
    for c in resp.metacog_assessment:
        name, conflict = normalize_label(c.rule_name, inst, maps)
        if audit:
            audit.claims += 1
            audit.conflicts += conflict
        key = (name, c.component)
        if name is None or key not in gt:
            continue
        if key in seen:
            if audit:
                audit.duplicates += 1
            continue
        seen.add(key)
        if audit:
            audit.mapped += 1
        truth = 1.0 if gt[key] else 0.0
        p_known = c.confidence if c.known else 1.0 - c.confidence
        sq.append((p_known - truth) ** 2)
    missing = len([k for k in gt if k not in seen])
    if audit:
        audit.unmatched_gt += missing
    sq += [0.25] * missing
    return max(0.0, 1.0 - sum(sq) / len(sq))


def normalized_ranking(resp: ParsedResponse, inst: Instance) -> List[str]:
    """Model ranking as internal hidden-rule names, first occurrence kept."""
    maps = label_maps(inst)
    hidden = set(maps[1].values())
    out: List[str] = []
    for item in resp.critical_unknowns_ranked:
        s = _repair(str(item))
        m = _H.search(s)
        name = maps[1].get(int(m.group(1))) if m else None
        if name is None:
            name, _ = normalize_label(s, inst, maps)
        if name in hidden and name not in out:
            out.append(name)
    return out


def attention_v2(resp: ParsedResponse, inst: Instance) -> float:
    truth = inst.true_unknown_ranking
    if not truth:
        return 1.0
    rank = {n: i for i, n in enumerate(truth)}
    filtered = [c for c in normalized_ranking(resp, inst) if c in rank]
    if len(filtered) < 2:
        if filtered and filtered[0] == truth[0]:
            return 0.6
        return 0.2 if filtered else 0.0
    conc = tot = 0
    for a in range(len(filtered)):
        for b in range(a + 1, len(filtered)):
            tot += 1
            conc += rank[filtered[a]] < rank[filtered[b]]
    return conc / tot


@dataclass
class ScoreBreakdownV2:
    objective: float
    calibration: float
    attention: float
    executive: float
    composite: float
    raw_objective: int
    best_objective: int
    worst_objective: int
    parse_errors: int
    scorer_version: str = SCORER_VERSION

    def to_dict(self) -> Dict:
        return asdict(self)


def score_response_v2(resp: ParsedResponse, inst: Instance, best_obj: int | None = None,
                      worst_obj: int | None = None, weights: Dict[str, float] | None = None,
                      audit: Optional[LabelAudit] = None) -> ScoreBreakdownV2:
    weights = weights or {"objective": 0.35, "calibration": 0.25, "attention": 0.2, "executive": 0.2}
    if best_obj is None:
        best_obj, _ = oracle_score(inst.world)
    if worst_obj is None:
        worst_obj = _worst_objective(inst.world)
    raw = run_plan(inst, resp.exploratory_actions, resp.final_plan).objective
    span = best_obj - worst_obj
    obj = 1.0 if span <= 0 else max(0.0, min(1.0, (raw - worst_obj) / span))
    cal = calibration_v2(resp, inst, audit)
    att = attention_v2(resp, inst)
    ex = _executive(resp, inst, best_obj=best_obj, worst_obj=worst_obj)
    comp = (weights["objective"] * obj + weights["calibration"] * cal
            + weights["attention"] * att + weights["executive"] * ex)
    return ScoreBreakdownV2(obj, cal, att, ex, comp, raw, best_obj, worst_obj, len(resp.errors))
